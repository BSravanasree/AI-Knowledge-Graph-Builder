"""
app.py — entry point.   Run:  streamlit run app.py

Pages
  Overview          architecture, KPIs, rebuild pipeline, run log
  Data sources (M1) connected sources, data-quality report, upload new docs
  Extraction (M2)   LLM / spaCy entity + relation extraction, live demo
  Ask (M3)          Graph-RAG question answering with cited evidence
  Graph explorer(M4) interactive graph with filters, search, drill-down
  Insights          fraud analytics charts
"""

import hmac

import streamlit as st

st.set_page_config(page_title="AI Knowledge Graph Builder", page_icon=":material/hub:", layout="wide")

from dashboard import (page_ask, page_explorer, page_extraction, page_insights,  # noqa: E402
                       page_overview, page_sources)
from dashboard.state import ensure_built, sidebar_status  # noqa: E402
from kg.config import get_settings  # noqa: E402


def password_gate() -> bool:
    """Optional: set APP_PASSWORD in secrets to restrict the deployed app."""
    pw = get_settings().app_password
    if not pw or st.session_state.get("authed"):
        return True
    st.title("AI Knowledge Graph Builder")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state.authed = True
            st.rerun()
        st.error("Incorrect password")
    return False


if password_gate():
    pages = [
        st.Page(page_overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(page_sources.render, title="Data sources", icon=":material/database:", url_path="sources"),
        st.Page(page_extraction.render, title="Extraction", icon=":material/account_tree:", url_path="extraction"),
        st.Page(page_ask.render, title="Ask (Graph-RAG)", icon=":material/forum:", url_path="ask"),
        st.Page(page_explorer.render, title="Graph explorer", icon=":material/hub:", url_path="explorer"),
        st.Page(page_insights.render, title="Insights", icon=":material/insights:", url_path="insights"),
    ]
    nav = st.navigation(pages)
    ensure_built()
    sidebar_status()
    nav.run()
