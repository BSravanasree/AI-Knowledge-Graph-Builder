import pandas as pd
import streamlit as st

from kg.config import get_settings
from kg.rag import answer_question

from .state import get_entity_names, get_index, get_store

EXAMPLES = [
    "Which merchant category has the highest fraud rate?",
    "Why is night-time risky, and what is being done about it?",
    "Who approved the Sentinel DeviceID agreement?",
    "What did Arjun Mehta find about Grocery fraud?",
    "Which risk factors have the biggest lift over the baseline fraud rate?",
    "What happened in incident INC-2041?",
]


def _md(text: str) -> str:
    """Escape $ so dollar amounts aren't rendered as LaTeX by Streamlit markdown."""
    return text.replace("\\$", "$").replace("$", "\\$")


def render():
    st.title("Ask the knowledge base")
    s = get_settings()
    st.caption("Milestone 3 — Graph-RAG: semantic search over documents, emails, case notes, transactions "
               "and pre-computed insights, combined with facts from the knowledge graph"
               + (" and answered by Gemini." if s.llm_enabled else ". (No LLM key: retrieval-only mode.)"))

    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.sidebar:
        st.caption("TRY A QUESTION")
        for q in EXAMPLES:
            if st.button(q, key=f"ex_{q}", width="stretch"):
                st.session_state.pending = q
        if st.session_state.chat and st.button("Clear conversation", width="stretch"):
            st.session_state.chat = []
            st.rerun()

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(_md(m["content"]))
            if m.get("answer"):
                _evidence(m["answer"])

    question = st.chat_input("Ask about fraud patterns, people, vendors, incidents…")
    question = question or st.session_state.pop("pending", None)
    if not question:
        if not st.session_state.chat:
            st.info("Pick an example question in the sidebar, or type your own below.")
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer…"):
            ans = answer_question(question, get_index(), get_store(), get_entity_names(), history=history)
        st.markdown(_md(ans.answer))
        _evidence(ans)
    st.session_state.chat.append({"role": "assistant", "content": ans.answer, "answer": ans})


def _evidence(ans):
    meta = f"{len(ans.sources)} passages · {len(ans.graph_facts)} graph facts · {ans.seconds}s"
    if ans.used_llm:
        meta += f" · {ans.model}"
    if ans.entities:
        meta += " · entities: " + ", ".join(ans.entities)
    with st.expander(f"Evidence ({meta})"):
        if ans.sources:
            st.dataframe(pd.DataFrame([{"#": i, "type": c["source_type"], "source": c["title"],
                                        "score": round(c["score"], 3), "text": c["text"]}
                                       for i, c in enumerate(ans.sources, 1)]),
                         hide_index=True, width="stretch")
        if ans.graph_facts:
            st.markdown("**Knowledge-graph facts**")
            st.markdown(_md("\n".join(f"- `G{i}` {f}" for i, f in enumerate(ans.graph_facts, 1))))
