"""Shared, cached resources for every dashboard page."""

from __future__ import annotations

import streamlit as st

from kg import schema
from kg.config import get_settings
from kg.graph_store import connect, last_connection_error
from kg.ingestion import load_case_database, load_transactions
from kg.rag import VectorIndex

ENTITY_NAME_LABELS = [l for l in schema.ALL_LABELS if l not in ("Transaction", "Document")]


@st.cache_resource(show_spinner="Connecting to the graph database…")
def get_store():
    return connect()


@st.cache_resource(show_spinner="Loading vector index…")
def get_index():
    return VectorIndex()


@st.cache_data(show_spinner=False)
def get_transactions():
    df, report = load_transactions(get_settings().transactions_csv)
    return df, report.as_dict()


@st.cache_data(show_spinner=False)
def get_cases():
    analysts, cases, report = load_case_database(get_settings().cases_db)
    return analysts, cases, report.as_dict()


@st.cache_data(show_spinner=False, ttl=300)
def get_stats():
    return get_store().stats()


@st.cache_data(show_spinner=False, ttl=300)
def get_entity_names():
    return get_store().all_names(ENTITY_NAME_LABELS)


def refresh_after_write(reload_index: bool = True):
    get_stats.clear()
    get_entity_names.clear()
    if reload_index:
        get_index.clear()


def ensure_built():
    """First run (or fresh cloud container): build the knowledge base automatically."""
    store, index = get_store(), get_index()
    if get_stats()["nodes"] > 0 and index.ready:
        return
    from kg.pipeline import run_full_build

    with st.status("First run: building the knowledge graph and vector index…", expanded=True) as status:
        bar = st.progress(0.0)
        line = st.empty()

        def progress(msg, frac):
            bar.progress(min(1.0, frac))
            line.write(msg)

        report = run_full_build(progress=progress, store=store)
        status.update(label=f"Knowledge base ready in {report.seconds}s", state="complete", expanded=False)
    refresh_after_write()


def sidebar_status():
    s = get_settings()
    store = get_store()
    with st.sidebar:
        st.caption("SYSTEM STATUS")
        if store.backend == "neo4j":
            st.markdown(":green[●] Graph: **Neo4j**")
        else:
            st.markdown(":orange[●] Graph: **local (NetworkX)**")
            err = last_connection_error()
            if err:
                st.caption(err[:160])
        st.markdown((":green[●] LLM: **Gemini**" if s.llm_enabled else ":orange[●] LLM: **off** (no API key)"))
        idx = get_index()
        emb = idx.meta.get("embedding_backend", "—") if idx.ready else "not built"
        st.markdown(f":green[●] Embeddings: **{emb.split(':')[0]}**")
