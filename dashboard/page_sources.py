import pandas as pd
import streamlit as st

from kg.ingestion import SUPPORTED_EXTENSIONS
from kg.pipeline import all_source_documents, ingest_uploaded_files

from .state import get_cases, get_index, get_store, get_transactions, refresh_after_write


def render():
    st.title("Data sources & ingestion")
    st.caption("Milestone 1 — connect, clean, validate and enrich enterprise data.")

    tx, tx_q = get_transactions()
    analysts, cases, db_q = get_cases()
    docs = all_source_documents()

    st.subheader("Connected sources")
    rows = [
        {"Source": "creditcard.csv", "Type": "Database table (CSV export)", "Records": f"{len(tx):,} transactions"},
        {"Source": "enterprise.db", "Type": "SQLite case-management DB",
         "Records": f"{len(cases)} fraud cases, {len(analysts)} analysts"},
    ]
    for d in docs:
        rows.append({"Source": d.path.split("/")[-1], "Type": d.source_type.upper(),
                     "Records": f"{len(d.text):,} characters"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Add documents (incremental graph update)")
    st.caption("Uploads are parsed, run through entity extraction, merged into the live graph and "
               "appended to the vector index — no full rebuild needed.")
    files = st.file_uploader("PDF, Word, Markdown, text or email files",
                             type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
                             accept_multiple_files=True)
    if files and st.button("Ingest uploaded files", type="primary"):
        with st.spinner("Parsing, extracting entities and updating the graph…"):
            rep = ingest_uploaded_files([(f.name, f.getvalue()) for f in files],
                                        store=get_store(), index=get_index())
        refresh_after_write(reload_index=False)
        for e in rep.errors:
            st.error(e)
        if rep.documents:
            st.success(f"Added {len(rep.documents)} document(s) in {rep.seconds}s.")
            st.dataframe(pd.DataFrame(rep.documents), hide_index=True, width="stretch")

    st.subheader("Data quality report")
    q = pd.DataFrame([{k: (v if k != "rule_failures" else (", ".join(f"{a}: {b}" for a, b in v.items()) or "none"))
                       for k, v in r.items()} for r in (tx_q, db_q)])
    st.dataframe(q, hide_index=True, width="stretch")
    st.caption("Validation rules: amount ≥ 0 · hour 0–23 · device trust 0–100 · velocity ≥ 0 · "
               "age 18–110 · binary flags are 0/1. Failing rows are rejected and counted.")

    tab1, tab2, tab3 = st.tabs(["Transactions (enriched)", "Fraud cases", "Document text"])
    with tab1:
        cols = ["tx_key", "amount", "transaction_hour", "time_slot", "merchant_category", "customer_segment",
                "device_trust_score", "velocity_last_24h", "risk_factors", "fraud_label"]
        only_fraud = st.toggle("Fraud only", value=False)
        view = tx[tx.is_fraud == 1] if only_fraud else tx
        st.dataframe(view[cols].head(500), hide_index=True, width="stretch")
    with tab2:
        st.dataframe(cases.drop(columns=["tx_key"], errors="ignore"), hide_index=True, width="stretch")
    with tab3:
        if docs:
            pick = st.selectbox("Document", docs, format_func=lambda d: f"{d.title}  ({d.source_type})")
            if pick.metadata:
                st.json(pick.metadata, expanded=False)
            st.text_area("Cleaned text", pick.text, height=320)
