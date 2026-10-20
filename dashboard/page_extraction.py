import pandas as pd
import streamlit as st

from kg.config import get_settings
from kg.extraction import extract
from kg.pipeline import all_source_documents, known_names

from .graph_view import render_graph
from .state import get_cases, get_transactions

SAMPLE = ("Harbor Trust Bank partnered with Sentinel DeviceID Ltd to roll out device fingerprinting in "
          "Mumbai. Priya Raman approved the rollout after the fraud team detected a rise in Location "
          "Mismatch cases at Travel merchants.")


def _show(g):
    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"**Entities** ({len(g.entities)})")
        st.dataframe(pd.DataFrame(g.entities)[["name", "label"]] if g.entities else pd.DataFrame(),
                     hide_index=True, width="stretch", height=300)
    with b:
        st.markdown(f"**Triples** ({len(g.relations)})")
        names = {e["key"]: e["name"] for e in g.entities}
        rows = [{"subject": names.get(r["src"], r["src"]), "relation": r["type"],
                 "object": names.get(r["dst"], r["dst"]), "evidence": r.get("evidence", "")} for r in g.relations]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=300)
    if g.entities:
        nodes = [{"key": e["key"], "name": e["name"], "label": e["label"]} for e in g.entities]
        edges = [{"src": r["src"], "dst": r["dst"], "type": r["type"], "props": {"evidence": r.get("evidence", "")}}
                 for r in g.relations]
        render_graph(nodes, edges, height=380)


def render():
    st.title("Entity & relationship extraction")
    s = get_settings()
    mode = "Gemini LLM (structured JSON output)" if s.llm_enabled and s.use_llm_extraction else "spaCy NER + dependency rules"
    st.caption(f"Milestone 2 — current extractor: **{mode}**, plus gazetteer linking to structured entities.")

    tx, _ = get_transactions()
    analysts, _, _ = get_cases()
    docs = all_source_documents()
    known = known_names(tx, analysts, docs)

    tab1, tab2 = st.tabs(["Try it on any text", "Per-document results"])
    with tab1:
        text = st.text_area("Paste text (an email, a paragraph from a report…)", SAMPLE, height=140)
        options = ["Auto"] + (["LLM"] if s.llm_enabled else []) + ["spaCy"]
        choice = st.segmented_control("Extractor", options, default="Auto")
        if text.strip():
            prefer = None if choice in (None, "Auto") else (choice == "LLM")
            with st.spinner("Extracting…"):
                g = extract(text, known, prefer_llm=prefer)
            st.caption(f"Method used: {g.method}")
            _show(g)
    with tab2:
        if not docs:
            st.info("No documents found.")
            return
        pick = st.selectbox("Document", docs, format_func=lambda d: f"{d.title}  ({d.source_type})")
        g = extract(pick.text, known)
        st.caption(f"Method: {g.method} · results are cached by content hash")
        _show(g)
