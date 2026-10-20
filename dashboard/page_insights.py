import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kg import schema

from .state import get_cases, get_transactions

BLUE = "#2a78d6"  # single-series colour (one hue — identity is carried by the axis labels)
INK = "#52514e"


def _bar(labels, values, texts, hover, title, axis_title, ref=None, ref_label="", horizontal=True):
    """
    Bar chart built from plain Python lists (str / float). Plotly 6+ ships numpy
    arrays in a binary format that some Streamlit versions don't decode, which
    showed up as a single unlabeled bar — plain lists render everywhere.
    """
    labels = [str(x) for x in labels]
    values = [float(v) for v in values]
    if horizontal:
        trace = go.Bar(x=values, y=labels, orientation="h", text=list(texts), hovertext=list(hover),
                       hoverinfo="text", marker_color=BLUE, textposition="outside", cliponaxis=False)
    else:
        trace = go.Bar(x=labels, y=values, text=list(texts), hovertext=list(hover),
                       hoverinfo="text", marker_color=BLUE, textposition="outside", cliponaxis=False)
    fig = go.Figure(trace)
    grid = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", title=axis_title)
    if horizontal:
        fig.update_layout(xaxis=grid, yaxis=dict(type="category", categoryorder="array", categoryarray=labels))
    else:
        fig.update_layout(yaxis=grid, xaxis=dict(type="category", categoryorder="array", categoryarray=labels))
    if ref is not None:
        line = dict(type="line", line=dict(color=INK, width=1, dash="dot"))
        if horizontal:
            fig.add_shape(**line, x0=ref, x1=ref, y0=0, y1=1, yref="paper")
            fig.add_annotation(x=ref, y=1.06, yref="paper", text=ref_label, showarrow=False,
                               font=dict(size=11, color=INK))
        else:
            fig.add_shape(**line, y0=ref, y1=ref, x0=0, x1=1, xref="paper")
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=320, bargap=0.35,
                      margin=dict(l=10, r=50, t=60, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def _rate_chart(df: pd.DataFrame, col: str, title: str, base: float, order=None):
    g = df.groupby(col).agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum")).reset_index()
    g["fraud_rate"] = 100 * g["fraud"] / g["transactions"]
    if order:
        g = g.set_index(col).reindex([o for o in order if o in set(g[col])]).reset_index()
    else:
        g = g.sort_values("fraud_rate")
    _bar(g[col].tolist(), g["fraud_rate"].tolist(),
         [f"{v:.2f}%" for v in g["fraud_rate"]],
         [f"{c}<br>{int(t):,} transactions · {int(f)} fraud · {r:.2f}%"
          for c, t, f, r in zip(g[col], g["transactions"], g["fraud"], g["fraud_rate"])],
         title, "Fraud rate (%)", ref=base, ref_label=f"portfolio {base:.2f}%")


def render():
    st.title("Fraud intelligence")
    st.caption("Analytics computed from the same cleaned data that feeds the graph — "
               "the patterns the RAG layer and documents talk about.")
    tx, _ = get_transactions()
    _, cases, _ = get_cases()
    base = 100 * tx["is_fraud"].mean()

    c = st.columns(4)
    c[0].metric("Transactions", f"{len(tx):,}")
    c[1].metric("Fraud cases", f"{int(tx.is_fraud.sum())}", f"{base:.2f}% of all", delta_color="off", delta_arrow="off")
    c[2].metric("Fraud value", f"${tx.loc[tx.is_fraud == 1, 'amount'].sum():,.0f}")
    c[3].metric("Open investigations", int((cases["status"] == "Open").sum()) if not cases.empty else 0)

    rows = []
    for rf in schema.RISK_FACTORS:
        m = tx["risk_factors"].map(lambda l: rf in l)
        rows.append({"risk_factor": rf, "transactions": int(m.sum()), "fraud": int(tx.loc[m, "is_fraud"].sum())})
    rf = pd.DataFrame(rows)
    rf["fraud_rate"] = 100 * rf["fraud"] / rf["transactions"]
    rf["lift"] = rf["fraud_rate"] / base
    rf = rf.sort_values("lift")
    _bar(rf["risk_factor"].tolist(), rf["lift"].tolist(), [f"{v:.1f}×" for v in rf["lift"]],
         [f"{n}<br>{int(t):,} transactions · {int(f)} fraud · {r:.2f}% · {l:.1f}× average"
          for n, t, f, r, l in zip(rf["risk_factor"], rf["transactions"], rf["fraud"], rf["fraud_rate"], rf["lift"])],
         "Risk-factor lift: how many times riskier than average", "Fraud rate ÷ portfolio rate",
         ref=1.0, ref_label="average")

    a, b = st.columns(2)
    with a:
        _rate_chart(tx, "merchant_category", "Fraud rate by merchant category", base)
    with b:
        _rate_chart(tx, "time_slot", "Fraud rate by time of day", base,
                    order=["Evening (18-23)", "Afternoon (12-17)", "Morning (06-11)", "Night (00-05)"])
    a, b = st.columns(2)
    with a:
        _rate_chart(tx, "customer_segment", "Fraud rate by customer age segment", base,
                    order=["Age 66+", "Age 51-65", "Age 36-50", "Age 26-35", "Age 18-25"])
    with b:
        g = tx.groupby("risk_factor_count").agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum")).reset_index()
        g["fraud_rate"] = 100 * g["fraud"] / g["n"]
        _bar([f"{int(k)} factors" for k in g["risk_factor_count"]], g["fraud_rate"].tolist(),
             [f"{v:.1f}%" for v in g["fraud_rate"]],
             [f"{int(k)} risk factors<br>{int(n):,} transactions · {int(f)} fraud"
              for k, n, f in zip(g["risk_factor_count"], g["n"], g["fraud"])],
             "Stacked risk: fraud rate by number of risk factors", "Fraud rate (%)", horizontal=False)

    with st.expander("Table view"):
        st.dataframe(rf.round(2), hide_index=True, width="stretch")
