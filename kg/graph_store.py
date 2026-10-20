"""
graph_store.py — Graph Construction & Storage Hub (Milestone 2).

Two interchangeable back-ends behind one interface:

  * Neo4jStore  — production store (Neo4j Desktop locally, Neo4j AuraDB in the
                  cloud). Batched UNWIND + MERGE writes, uniqueness constraint
                  on node keys, so re-running the pipeline is idempotent and
                  new data is merged in incrementally.
  * LocalStore  — NetworkX graph persisted to storage/graph.pkl. Used when
                  Neo4j isn't configured/reachable, so the app never breaks
                  during a demo.

Every node carries: key (unique id), name, label (+ the Neo4j label itself).
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Optional

from . import schema
from .config import get_settings


def _chunks(rows: list, size: int) -> Iterable[list]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _clean_props(d: dict) -> dict:
    """Neo4j properties must be primitives or lists of primitives."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if hasattr(v, "item"):  # numpy scalar
            v = v.item()
        if isinstance(v, (list, tuple)):
            v = [x.item() if hasattr(x, "item") else x for x in v]
        if isinstance(v, (str, int, float, bool, list)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class GraphStore(ABC):
    backend = "base"

    # ── writes ───────────────────────────────────────────────────────────────
    @abstractmethod
    def ensure_schema(self): ...

    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def upsert_nodes(self, label: str, rows: list[dict]): ...

    @abstractmethod
    def upsert_edges(self, rel: str, rows: list[dict]): ...

    def save(self):
        """Persist (no-op for Neo4j, which persists on write)."""

    # ── reads ────────────────────────────────────────────────────────────────
    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def seed_nodes(self, search: str, labels: list[str], fraud_only: bool, limit: int) -> list[dict]: ...

    @abstractmethod
    def expand(self, keys: list[str], labels: list[str], fraud_only: bool, limit: int) -> tuple[list[dict], list[dict]]: ...

    @abstractmethod
    def get_node(self, key: str) -> Optional[dict]: ...

    @abstractmethod
    def node_facts(self, key: str, limit: int = 20) -> dict: ...

    @abstractmethod
    def top_degree(self, exclude: list[str], n: int = 10) -> list[dict]: ...

    @abstractmethod
    def all_names(self, labels: list[str]) -> list[dict]: ...

    # ── shared logic ─────────────────────────────────────────────────────────
    def subgraph(self, search: str = "", labels: Optional[list[str]] = None, fraud_only: bool = True,
                 max_nodes: int = 150, depth: int = 1) -> tuple[list[dict], list[dict]]:
        """Seed nodes matching `search`, then expand `depth` hops — bounded by max_nodes."""
        labels = labels or []
        seeds = self.seed_nodes(search.strip().lower(), labels if not search else [], fraud_only,
                                limit=max(10, max_nodes // 3) if search else max_nodes)
        nodes = {n["key"]: n for n in seeds}
        edges: dict[tuple, dict] = {}
        frontier = list(nodes)
        for _ in range(max(1, depth)):
            if not frontier or len(nodes) >= max_nodes:
                break
            new_nodes, new_edges = self.expand(frontier, labels, fraud_only, limit=max_nodes * 4)
            frontier = []
            for n in new_nodes:
                if n["key"] not in nodes and len(nodes) < max_nodes:
                    nodes[n["key"]] = n
                    frontier.append(n["key"])
            for e in new_edges:
                if e["src"] in nodes and e["dst"] in nodes:
                    edges[(e["src"], e["type"], e["dst"])] = e
        # edges among already-present nodes (seed↔seed)
        _, extra = self.expand(list(nodes), labels, fraud_only, limit=5000)
        for e in extra:
            if e["src"] in nodes and e["dst"] in nodes:
                edges[(e["src"], e["type"], e["dst"])] = e
        return list(nodes.values()), list(edges.values())

    def facts_text(self, keys: list[str], per_node: int = 12) -> list[str]:
        """Human-readable facts about nodes — the graph half of Graph-RAG."""
        lines = []
        for k in keys:
            f = self.node_facts(k, limit=per_node)
            if not f:
                continue
            n = f["node"]
            head = f"{n.get('name')} ({n.get('label')})"
            if n.get("label") == "Transaction" and n.get("text"):
                lines.append(n["text"])
            elif n.get("description"):
                lines.append(f"{head}: {n['description']}")
            if f["tx_total"]:
                if n.get("label") in ("MerchantCategory", "TimeSlot", "CustomerSegment", "RiskFactor"):
                    rate = 100 * f["tx_fraud"] / f["tx_total"]
                    lines.append(f"{head} is linked to {f['tx_total']} transactions, "
                                 f"{f['tx_fraud']} of them fraud ({rate:.2f}% fraud rate).")
                else:
                    lines.append(f"{head} is connected to {f['tx_total']} transactions "
                                 f"({f['tx_fraud']} flagged as fraud).")
            for r in f["relations"]:
                if r["out"]:
                    lines.append(f"{head} -[{r['type']}]-> {r['name']} ({r['label']})")
                else:
                    lines.append(f"{r['name']} ({r['label']}) -[{r['type']}]-> {head}")
        return lines


# ═════════════════════════════════════════════════════════════════════════════
# Neo4j
# ═════════════════════════════════════════════════════════════════════════════
class Neo4jStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = ""):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database or None

    def _run(self, query: str, **params) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            return [r.data() for r in session.run(query, **params)]

    def close(self):
        self.driver.close()

    def ensure_schema(self):
        self._run("CREATE CONSTRAINT node_key IF NOT EXISTS FOR (n:Node) REQUIRE n.key IS UNIQUE")
        self._run("CREATE INDEX node_name IF NOT EXISTS FOR (n:Node) ON (n.name)")
        self._run("CREATE INDEX node_label IF NOT EXISTS FOR (n:Node) ON (n.label)")

    def clear(self):
        while True:
            res = self._run("MATCH (n:Node) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c")
            if not res or res[0]["c"] == 0:
                break

    def upsert_nodes(self, label: str, rows: list[dict]):
        if label not in schema.ALL_LABELS:  # labels can't be parameters -> allowlist
            raise ValueError(f"Unknown label {label}")
        q = (f"UNWIND $rows AS row MERGE (n:Node {{key: row.key}}) "
             f"SET n += row, n.label = '{label}', n:`{label}`")
        for batch in _chunks([_clean_props(r) for r in rows], 1000):
            self._run(q, rows=batch)

    def upsert_edges(self, rel: str, rows: list[dict]):
        rel = schema.normalise_relation(rel)  # A-Z0-9_ only -> injection-safe
        q = (f"UNWIND $rows AS row "
             f"MATCH (a:Node {{key: row.src}}) MATCH (b:Node {{key: row.dst}}) "
             f"MERGE (a)-[r:`{rel}`]->(b) SET r += row.props")
        payload = [{"src": r["src"], "dst": r["dst"], "props": _clean_props(r.get("props", {}))} for r in rows]
        for batch in _chunks(payload, 1000):
            self._run(q, rows=batch)

    def stats(self) -> dict:
        labels = {r["label"]: r["c"] for r in self._run(
            "MATCH (n:Node) RETURN n.label AS label, count(*) AS c")}
        rels = {r["t"]: r["c"] for r in self._run(
            "MATCH (:Node)-[r]->(:Node) RETURN type(r) AS t, count(*) AS c")}
        return {"nodes": sum(labels.values()), "relationships": sum(rels.values()),
                "labels": labels, "rel_types": rels}

    _FILTER = ("(size($labels) = 0 OR {v}.label IN $labels) AND "
               "(NOT $fraud_only OR {v}.label <> 'Transaction' OR {v}.is_fraud = 1)")

    def seed_nodes(self, search, labels, fraud_only, limit):
        q = ("MATCH (n:Node) WHERE " + self._FILTER.format(v="n") +
             " AND ($q = '' OR toLower(n.name) CONTAINS $q OR toLower(n.key) CONTAINS $q) "
             "RETURN n {.*} AS n ORDER BY CASE WHEN n.label = 'Transaction' THEN 1 ELSE 0 END LIMIT $limit")
        return [r["n"] for r in self._run(q, labels=labels, fraud_only=fraud_only, q=search, limit=limit)]

    def expand(self, keys, labels, fraud_only, limit):
        q = ("MATCH (n:Node)-[r]-(m:Node) WHERE n.key IN $keys AND " + self._FILTER.format(v="m") +
             " RETURN startNode(r).key AS src, type(r) AS type, endNode(r).key AS dst, "
             "properties(r) AS props, m {.*} AS m LIMIT $limit")
        rows = self._run(q, keys=keys, labels=labels, fraud_only=fraud_only, limit=limit)
        nodes = {r["m"]["key"]: r["m"] for r in rows}
        edges = [{"src": r["src"], "dst": r["dst"], "type": r["type"], "props": r["props"]} for r in rows]
        return list(nodes.values()), edges

    def get_node(self, key):
        res = self._run("MATCH (n:Node {key: $k}) RETURN n {.*} AS n", k=key)
        return res[0]["n"] if res else None

    def node_facts(self, key, limit=20):
        node = self.get_node(key)
        if not node:
            return {}
        rels = self._run(
            "MATCH (n:Node {key: $k})-[r]-(m:Node) WHERE m.label <> 'Transaction' "
            "RETURN startNode(r) = n AS out, type(r) AS type, m.name AS name, m.label AS label, m.key AS key "
            "LIMIT $limit", k=key, limit=limit)
        tx = self._run(
            "MATCH (n:Node {key: $k})--(t:Node {label: 'Transaction'}) "
            "RETURN count(DISTINCT t) AS total, coalesce(sum(t.is_fraud), 0) AS fraud", k=key)
        total = tx[0]["total"] if tx and node.get("label") != "Transaction" else 0
        fraud = tx[0]["fraud"] if tx and node.get("label") != "Transaction" else 0
        return {"node": node, "relations": rels, "tx_total": total, "tx_fraud": fraud}

    def top_degree(self, exclude, n=10):
        return self._run(
            "MATCH (n:Node) WHERE NOT n.label IN $exclude "
            "RETURN n.key AS key, n.name AS name, n.label AS label, size([(n)--() | 1]) AS degree "
            "ORDER BY degree DESC LIMIT $n", exclude=exclude, n=n)

    def all_names(self, labels):
        return self._run("MATCH (n:Node) WHERE n.label IN $labels RETURN n.key AS key, n.name AS name, "
                         "n.label AS label", labels=labels)


# ═════════════════════════════════════════════════════════════════════════════
# Local (NetworkX) fallback
# ═════════════════════════════════════════════════════════════════════════════
class LocalStore(GraphStore):
    backend = "local"

    def __init__(self, path=None):
        import networkx as nx

        self.path = path or (get_settings().storage_dir / "graph.pkl")
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.g = pickle.load(f)
        else:
            self.g = nx.MultiDiGraph()

    def ensure_schema(self):
        pass

    def clear(self):
        self.g.clear()
        self.save()

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.g, f, protocol=pickle.HIGHEST_PROTOCOL)

    def upsert_nodes(self, label, rows):
        for r in rows:
            props = _clean_props(r)
            props["label"] = label
            if self.g.has_node(r["key"]):
                self.g.nodes[r["key"]].update(props)
            else:
                self.g.add_node(r["key"], **props)

    def upsert_edges(self, rel, rows):
        rel = schema.normalise_relation(rel)
        for r in rows:
            if not (self.g.has_node(r["src"]) and self.g.has_node(r["dst"])):
                continue
            existing = self.g.get_edge_data(r["src"], r["dst"], key=rel)
            props = _clean_props(r.get("props", {}))
            if existing is not None:
                existing.update(props)
            else:
                self.g.add_edge(r["src"], r["dst"], key=rel, type=rel, **props)

    def stats(self):
        labels = Counter(d.get("label") for _, d in self.g.nodes(data=True))
        rels = Counter(k for _, _, k in self.g.edges(keys=True))
        return {"nodes": self.g.number_of_nodes(), "relationships": self.g.number_of_edges(),
                "labels": dict(labels), "rel_types": dict(rels)}

    def _ok(self, d, labels, fraud_only):
        if labels and d.get("label") not in labels:
            return False
        if fraud_only and d.get("label") == "Transaction" and d.get("is_fraud") != 1:
            return False
        return True

    def _node(self, k):
        d = dict(self.g.nodes[k])
        d["key"] = k
        return d

    def seed_nodes(self, search, labels, fraud_only, limit):
        out = []
        ordered = sorted(self.g.nodes(data=True), key=lambda kv: kv[1].get("label") == "Transaction")
        for k, d in ordered:
            if not self._ok(d, labels, fraud_only):
                continue
            if search and search not in str(d.get("name", "")).lower() and search not in k.lower():
                continue
            out.append(self._node(k))
            if len(out) >= limit:
                break
        return out

    def expand(self, keys, labels, fraud_only, limit):
        nodes, edges = {}, []
        for k in keys:
            if not self.g.has_node(k):
                continue
            for src, dst, rel, data in list(self.g.out_edges(k, keys=True, data=True)) + \
                    list(self.g.in_edges(k, keys=True, data=True)):
                other = dst if src == k else src
                if not self._ok(self.g.nodes[other], labels, fraud_only):
                    continue
                nodes[other] = self._node(other)
                edges.append({"src": src, "dst": dst, "type": rel,
                              "props": {kk: vv for kk, vv in data.items() if kk != "type"}})
                if len(edges) >= limit:
                    return list(nodes.values()), edges
        return list(nodes.values()), edges

    def get_node(self, key):
        return self._node(key) if self.g.has_node(key) else None

    def node_facts(self, key, limit=20):
        if not self.g.has_node(key):
            return {}
        node = self._node(key)
        rels, total, fraud = [], 0, 0
        for src, dst, rel in list(self.g.out_edges(key, keys=True)) + list(self.g.in_edges(key, keys=True)):
            other = dst if src == key else src
            od = self.g.nodes[other]
            if od.get("label") == "Transaction":
                if node.get("label") != "Transaction":
                    total += 1
                    fraud += int(od.get("is_fraud", 0))
                continue
            if len(rels) < limit:
                rels.append({"out": src == key, "type": rel, "name": od.get("name"),
                             "label": od.get("label"), "key": other})
        return {"node": node, "relations": rels, "tx_total": total, "tx_fraud": fraud}

    def top_degree(self, exclude, n=10):
        rows = [{"key": k, "name": d.get("name"), "label": d.get("label"), "degree": self.g.degree(k)}
                for k, d in self.g.nodes(data=True) if d.get("label") not in exclude]
        return sorted(rows, key=lambda r: -r["degree"])[:n]

    def all_names(self, labels):
        return [{"key": k, "name": d.get("name"), "label": d.get("label")}
                for k, d in self.g.nodes(data=True) if d.get("label") in labels]


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════
_last_error = ""


def connect() -> GraphStore:
    """GRAPH_BACKEND=auto (default) uses Neo4j when configured & reachable, else local."""
    global _last_error
    s = get_settings()
    want = s.graph_backend.lower()
    if want in ("auto", "neo4j") and s.neo4j_configured:
        try:
            store = Neo4jStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database)
            store.ensure_schema()
            _last_error = ""
            return store
        except Exception as exc:
            _last_error = f"Neo4j unavailable ({exc.__class__.__name__}: {exc})"
            print(f"[graph_store] {_last_error} — using local graph instead.")
            if want == "neo4j":
                raise
    return LocalStore()


def last_connection_error() -> str:
    return _last_error
