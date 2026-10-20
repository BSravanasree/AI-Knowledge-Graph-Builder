import pandas as pd
import streamlit as st

from kg.config import get_settings
from kg.pipeline import read_run_log, run_full_build

from .state import get_index, get_stats, get_store, get_transactions, refresh_after_write

STAGES = [
    ("Sources", "Transactions table · case DB · PDF / DOCX / MD / TXT · emails"),
    ("M1 Ingestion", "clean · validate · enrich · schema"),
    ("M2 Extraction", "Gemini LLM → spaCy fallback · entity resolution"),
    ("Storage", "Neo4j knowledge graph + FAISS vector index"),
    ("M3 Graph-RAG", "hybrid retrieval + graph facts → cited answer"),
    ("M4 Dashboard", "explore · search · ask · upload"),
]


def _architecture_html() -> str:
    cards = []
    for i, (title, body) in enumerate(STAGES):
        cards.append(f'<div class="kg-card"><b>{title}</b><span>{body}</span></div>')
        if i < len(STAGES) - 1:
            cards.append('<div class="kg-arrow">→</div>')
    return ("<style>.kg-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;margin:4px 0 12px}"
            ".kg-card{flex:1 1 140px;border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px;"
            "display:flex;flex-direction:column;gap:4px;font-size:13px;line-height:1.35}"
            ".kg-card span{opacity:.75}.kg-arrow{display:flex;align-items:center;opacity:.5;font-size:18px}"
            "@media (max-width:700px){.kg-arrow{display:none}}</style>"
            f'<div class="kg-flow">{"".join(cards)}</div>')


def render():
    st.title("AI Knowledge Graph Builder")
    st.caption("Enterprise intelligence over transactions, case records, documents and emails — "
               "graph construction, Graph-RAG search and interactive exploration.")

    stats = get_stats()
    tx, _ = get_transactions()
    idx = get_index()
    labels = stats.get("labels", {})
    c = st.columns(5)
    c[0].metric("Graph nodes", f"{stats['nodes']:,}")
    c[1].metric("Relationships", f"{stats['relationships']:,}")
    c[2].metric("Documents & emails", labels.get("Document", 0))
    c[3].metric("Extracted entities", sum(labels.get(l, 0) for l in
                                          ("Person", "Organization", "Location", "Product", "Policy", "Event", "Concept")))
    c[4].metric("Fraud rate", f"{100 * tx['is_fraud'].mean():.2f}%", help=f"{int(tx['is_fraud'].sum())} of {len(tx):,} transactions")

    st.subheader("Architecture")
    st.html(_architecture_html())

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Graph composition")
        comp = pd.DataFrame(sorted(labels.items(), key=lambda kv: -kv[1]), columns=["Node label", "Count"])
        rels = pd.DataFrame(sorted(stats.get("rel_types", {}).items(), key=lambda kv: -kv[1]),
                            columns=["Relationship", "Count"])
        a, b = st.columns(2)
        a.dataframe(comp, hide_index=True, width="stretch")
        b.dataframe(rels, hide_index=True, width="stretch")
    with right:
        st.subheader("Pipeline")
        st.write(f"Graph backend: **{get_store().backend}** · Vector index: "
                 f"**{idx.meta.get('chunks', 0):,} chunks** ({idx.meta.get('embedding_backend', '—')})")
        clear = st.checkbox("Clear graph before rebuilding", value=True)
        if st.button("Rebuild knowledge base", type="primary", width="stretch"):
            bar, msg = st.progress(0.0), st.empty()
            rep = run_full_build(progress=lambda m, f: (bar.progress(min(1.0, f)), msg.write(m)),
                                 clear=clear, store=get_store())
            refresh_after_write()
            st.success(f"Rebuilt in {rep.seconds}s — {rep.graph['nodes']:,} nodes, "
                       f"{rep.graph['relationships']:,} relationships.")
        runs = read_run_log(5)
        if runs:
            st.caption("Recent runs")
            st.dataframe(pd.DataFrame([{"started": r["started"], "type": r["kind"], "backend": r["backend"],
                                        "seconds": r["seconds"], "errors": len(r.get("errors", []))}
                                       for r in runs]), hide_index=True, width="stretch")
            with st.expander("Last run — step timings"):
                st.dataframe(pd.DataFrame(runs[0]["steps"]), hide_index=True, width="stretch")

    if not get_settings().llm_enabled:
        st.info("Add a `GEMINI_API_KEY` to enable LLM-based entity extraction and generated answers. "
                "Without it the app uses spaCy extraction and retrieval-only search.")
