import pandas as pd
import streamlit as st

from kg import schema

from .graph_view import render_graph
from .state import get_stats, get_store

PRESETS = {
    "Fraud network (default)": dict(search="", labels=[], fraud_only=True),
    "Who's who — people, organisations, documents": dict(
        search="", labels=["Person", "Organization", "Location", "Document", "Product", "Policy", "Event"],
        fraud_only=True),
    "Risk drivers": dict(search="", labels=["RiskFactor", "MerchantCategory", "TimeSlot", "Transaction",
                                            "Document"], fraud_only=True),
}


def render():
    st.title("Knowledge graph explorer")
    st.caption("Milestone 4 — filter, search and drill down. Hover a node for its properties; "
               "drag to rearrange; scroll to zoom.")
    store = get_store()
    stats = get_stats()
    present = [l for l in schema.ALL_LABELS if stats["labels"].get(l)]

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 3, 2])
        preset = c1.selectbox("View", list(PRESETS))
        p = PRESETS[preset]
        search = c2.text_input("Search nodes", value=p["search"],
                               placeholder="e.g. Grocery, Arjun Mehta, Sentinel, TX_2051")
        labels = c3.multiselect("Node types", present, default=[l for l in p["labels"] if l in present])
        c4, c5, c6, c7 = st.columns(4)
        depth = c4.slider("Hops from matches", 1, 3, 1 if not search else 2)
        max_nodes = c5.slider("Max nodes", 20, 400, 80, step=10)
        fraud_only = c6.toggle("Only fraud transactions", value=p["fraud_only"],
                               help="10,000 transaction nodes would hide everything else; "
                                    "by default only the 151 fraud transactions are drawn.")
        edge_labels = c7.toggle("Show relationship labels", value=False,
                                help="Hover an edge to see its relationship type either way.")

    nodes, edges = store.subgraph(search=search, labels=labels, fraud_only=fraud_only,
                                  max_nodes=max_nodes, depth=depth)
    if not nodes:
        st.warning("No nodes match. Try another search or clear the type filter.")
        return
    highlight = nodes[0]["key"] if search else ""
    render_graph(nodes, edges, height=660, highlight=highlight, edge_labels=edge_labels)
    st.caption(f"Showing {len(nodes)} nodes and {len(edges)} relationships "
               f"(graph total: {stats['nodes']:,} / {stats['relationships']:,}).")

    st.subheader("Drill down")
    options = sorted(nodes, key=lambda n: (n.get("label") == "Transaction", str(n.get("name"))))
    pick = st.selectbox("Node", options, format_func=lambda n: f"{n.get('name')}  ·  {n.get('label')}")
    if pick:
        facts = store.node_facts(pick["key"], limit=50)
        a, b = st.columns([2, 3])
        with a:
            props = {k: v for k, v in facts["node"].items() if k not in ("content_hash",)}
            st.json(props, expanded=True)
        with b:
            if facts["tx_total"]:
                rate = 100 * facts["tx_fraud"] / facts["tx_total"]
                st.metric("Connected transactions", f"{facts['tx_total']:,}",
                          f"{facts['tx_fraud']} fraud · {rate:.2f}%", delta_color="off")
            if facts["relations"]:
                st.dataframe(pd.DataFrame([{
                    "direction": "→" if r["out"] else "←", "relationship": r["type"],
                    "node": r["name"], "type": r["label"]} for r in facts["relations"]]),
                    hide_index=True, width="stretch")

    with st.expander("Most connected entities"):
        top = store.top_degree(exclude=["Transaction", "FraudStatus"], n=15)
        st.dataframe(pd.DataFrame(top)[["name", "label", "degree"]], hide_index=True, width="stretch")
