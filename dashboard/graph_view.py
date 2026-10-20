"""Interactive graph rendering with vis-network (bundled locally — works offline)."""

from __future__ import annotations

import html
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import streamlit.components.v1 as components

from kg.schema import LABEL_COLORS

_STATIC = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def _vis_js() -> str:
    return (_STATIC / "vis-network.min.js").read_text(encoding="utf-8")


def _tooltip(n: dict) -> str:
    skip = {"key", "text", "description", "content_hash", "name", "label"}
    lines = [f"{n.get('name')}  [{n.get('label')}]"]
    if n.get("description"):
        lines.append(str(n["description"])[:200])
    for k, v in n.items():
        if k not in skip and v not in (None, "", []):
            lines.append(f"{k}: {v}")
    return "\n".join(lines[:12])


def render_graph(nodes: list[dict], edges: list[dict], height: int = 640, highlight: str = "",
                 edge_labels: bool = False):
    degree = Counter()
    for e in edges:
        degree[e["src"]] += 1
        degree[e["dst"]] += 1

    vis_nodes = []
    for n in nodes:
        label = n.get("label", "Concept")
        color = LABEL_COLORS.get(label, "#adb5bd")
        if label == "Transaction" and n.get("is_fraud") == 1:
            color = "#d62828"
        name = str(n.get("name", n["key"]))
        vis_nodes.append({
            "id": n["key"],
            # transaction ids are noise at a glance — they show on hover instead
            "label": "" if label == "Transaction" else (name if len(name) <= 28 else name[:26] + "…"),
            "title": _tooltip(n),
            "group": label,
            "color": {"background": color, "border": "#1d3557" if n["key"] == highlight else color,
                      "highlight": {"background": color, "border": "#1d3557"}},
            "borderWidth": 4 if n["key"] == highlight else 1,
            "size": 8 + min(26, degree[n["key"]] ** 0.6 * 3) if label != "Transaction" else 7,
            "shape": "dot" if label != "Document" else "square",
            "font": {"size": 11 if label == "Transaction" else 14},
        })
    vis_edges = []
    for e in edges:
        rel = e["type"].replace("_", " ").lower()
        evidence = e["props"].get("evidence", "") if e.get("props") else ""
        item = {"from": e["src"], "to": e["dst"], "arrows": "to",
                "title": rel + (f"\n{evidence}" if evidence else "")}
        if edge_labels:
            item["label"] = rel
        vis_edges.append(item)

    legend = "".join(
        f'<span class="chip"><i style="background:{LABEL_COLORS.get(l, "#adb5bd")}"></i>{html.escape(l)} {c}</span>'
        for l, c in Counter(n.get("label") for n in nodes).most_common())

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<script>{_vis_js()}</script>
<style>
  body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: transparent; }}
  #net {{ height:{height - 40}px; border:1px solid #d9dee5; border-radius:10px; background:#fbfcfd; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:6px 12px; font-size:12px; color:#3d4a5c; padding:6px 2px 8px; }}
  .chip i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px; }}
  div.vis-tooltip {{ white-space: pre-wrap; max-width: 360px; font-size: 12px; }}
</style></head><body>
<div class="legend">{legend}</div>
<div id="net"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(vis_nodes)});
  const edges = new vis.DataSet({json.dumps(vis_edges)});
  const net = new vis.Network(document.getElementById("net"), {{nodes, edges}}, {{
    physics: {{ solver: "forceAtlas2Based", forceAtlas2Based: {{ gravitationalConstant: -45, springLength: 90 }},
               stabilization: {{ iterations: 250 }} }},
    edges: {{ font: {{ size: 9, color: "#6c7a89", strokeWidth: 0 }}, color: {{ color: "#c9d1d9", highlight: "#1d3557", hover: "#1d3557" }}, width: 0.6,
             smooth: {{ type: "continuous" }}, arrows: {{ to: {{ scaleFactor: 0.4 }} }} }},
    interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: false }}
  }});
  net.once("stabilizationIterationsDone", () => net.setOptions({{ physics: false }}));
</script></body></html>"""
    components.html(doc, height=height, scrolling=False)
