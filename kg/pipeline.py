"""
pipeline.py — end-to-end ETL orchestration (M1 → M2 → M3).

    Extract    : transactions CSV, case DB, documents, emails      (ingestion.py)
    Transform  : clean, validate, enrich; LLM/NLP entity extraction (extraction.py)
    Load       : knowledge graph (graph_store.py) + vector index    (rag.py)

Two entry points:
    run_full_build()        — rebuild everything from the configured sources
    ingest_uploaded_files() — incremental: add new documents to the live graph
                              and vector index without a rebuild

Every run is appended to storage/run_log.jsonl (simple pipeline monitoring).
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from email.utils import getaddresses
from typing import Callable, Optional

import pandas as pd

from . import schema
from .config import get_settings
from .extraction import ExtractedGraph, canonical_name, extract
from .graph_store import GraphStore, connect
from .ingestion import (
    SourceDocument, load_case_database, load_documents, load_transactions,
    parse_file, transaction_to_text,
)
from .rag import VectorIndex

Progress = Optional[Callable[[str, float], None]]


@dataclass
class BuildReport:
    started: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    kind: str = "full"
    steps: list = field(default_factory=list)
    quality: list = field(default_factory=list)
    documents: list = field(default_factory=list)
    graph: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    backend: str = ""
    seconds: float = 0.0
    errors: list = field(default_factory=list)

    def step(self, name: str, t0: float, detail: str = ""):
        self.steps.append({"step": name, "seconds": round(time.time() - t0, 2), "detail": detail})

    def as_dict(self):
        return self.__dict__


def _log_run(report: BuildReport):
    p = get_settings().storage_dir / "run_log.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(report.as_dict(), default=str) + "\n")


def read_run_log(n: int = 10) -> list[dict]:
    p = get_settings().storage_dir / "run_log.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(x) for x in lines[-n:]][::-1]


# ═════════════════════════════════════════════════════════════════════════════
# Structured graph (transactions + case DB)
# ═════════════════════════════════════════════════════════════════════════════
def build_structured_graph(store: GraphStore, tx: pd.DataFrame, analysts: pd.DataFrame,
                           cases: pd.DataFrame) -> dict:
    mk = schema.make_key
    # dimension nodes
    store.upsert_nodes("MerchantCategory", [{"key": mk("MerchantCategory", m), "name": m}
                                            for m in sorted(tx["merchant_category"].unique())])
    store.upsert_nodes("TimeSlot", [{"key": mk("TimeSlot", t), "name": t} for t in sorted(tx["time_slot"].unique())])
    store.upsert_nodes("CustomerSegment", [{"key": mk("CustomerSegment", c), "name": c}
                                           for c in sorted(tx["customer_segment"].unique())])
    store.upsert_nodes("RiskFactor", [{"key": mk("RiskFactor", r), "name": r, "description": d}
                                      for r, d in schema.RISK_FACTORS.items()])
    store.upsert_nodes("FraudStatus", [{"key": mk("FraudStatus", s), "name": s} for s in ("Fraud", "Normal")])

    # transactions
    tx_rows = []
    for r in tx.to_dict("records"):
        tx_rows.append({
            "key": r["tx_key"], "name": r["tx_key"], "transaction_id": r["transaction_id"],
            "amount": round(float(r["amount"]), 2), "hour": r["transaction_hour"],
            "merchant_category": r["merchant_category"], "is_fraud": r["is_fraud"],
            "fraud_label": r["fraud_label"], "device_trust_score": r["device_trust_score"],
            "velocity_last_24h": r["velocity_last_24h"], "cardholder_age": r["cardholder_age"],
            "risk_factors": r["risk_factors"], "risk_factor_count": r["risk_factor_count"],
            "text": transaction_to_text(r), "source": "creditcard.csv",
        })
    store.upsert_nodes("Transaction", tx_rows)

    edges = defaultdict(list)
    for r in tx.to_dict("records"):
        k = r["tx_key"]
        edges["AT_MERCHANT"].append({"src": k, "dst": mk("MerchantCategory", r["merchant_category"])})
        edges["IN_TIMESLOT"].append({"src": k, "dst": mk("TimeSlot", r["time_slot"])})
        edges["BY_SEGMENT"].append({"src": k, "dst": mk("CustomerSegment", r["customer_segment"])})
        edges["FLAGGED_AS"].append({"src": k, "dst": mk("FraudStatus", r["fraud_label"])})
        for f in r["risk_factors"]:
            edges["HAS_RISK_FACTOR"].append({"src": k, "dst": mk("RiskFactor", f)})

    # case-management DB: analysts investigate fraud transactions
    if not analysts.empty:
        locs = sorted(analysts["location"].dropna().unique())
        store.upsert_nodes("Location", [{"key": mk("Location", l), "name": l} for l in locs])
        store.upsert_nodes("Person", [{
            "key": mk("Person", a["name"]), "name": a["name"], "team": a["team"],
            "description": f"{a['name']} — {a['team']} analyst based in {a['location']}.",
            "source": "enterprise.db"} for a in analysts.to_dict("records")])
        for a in analysts.to_dict("records"):
            edges["LOCATED_IN"].append({"src": mk("Person", a["name"]), "dst": mk("Location", a["location"])})
    for c in cases.to_dict("records") if not cases.empty else []:
        edges["INVESTIGATES"].append({"src": mk("Person", c["analyst"]), "dst": c["tx_key"],
                                      "props": {"case_id": c["case_id"], "status": c["status"],
                                                "opened_on": c["opened_on"]}})
    for rel, rows in edges.items():
        store.upsert_edges(rel, rows)
    return {rel: len(rows) for rel, rows in edges.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Unstructured graph (documents + emails)
# ═════════════════════════════════════════════════════════════════════════════
_NON_PERSON = re.compile(r"\b(team|staff|all|group|support|noreply|desk)\b", re.I)


def email_people(doc: SourceDocument) -> tuple[list[str], list[str]]:
    def names(field_value):
        out = []
        for display, _addr in getaddresses([field_value or ""]):
            display = display.strip()
            if len(display.split()) >= 2 and not _NON_PERSON.search(display):
                out.append(display)
        return out

    return names(doc.metadata.get("from")), names(doc.metadata.get("to"))


def known_names(tx: pd.DataFrame, analysts: pd.DataFrame, docs: list[SourceDocument]) -> dict[str, str]:
    """Names we already know from structured data → label (the gazetteer)."""
    known = {m: "MerchantCategory" for m in tx["merchant_category"].unique()}
    known.update({r: "RiskFactor" for r in schema.RISK_FACTORS})
    if not analysts.empty:
        known.update({n: "Person" for n in analysts["name"]})
        known.update({l: "Location" for l in analysts["location"].dropna().unique()})
    for d in docs:
        if d.source_type == "email":
            frm, to = email_people(d)
            known.update({n: "Person" for n in frm + to})
    return known


def resolve_entities(results: dict[str, ExtractedGraph], known: dict[str, str]) -> dict[str, ExtractedGraph]:
    """
    Global entity resolution across all documents:
      * one label per entity name (structured/gazetteer label wins, otherwise majority vote)
      * short org names merge into the unique longer name ("Sentinel" -> "Sentinel DeviceID")
    """
    known_lc = {canonical_name(k).lower(): v for k, v in known.items()}
    votes: dict[str, Counter] = defaultdict(Counter)
    display: dict[str, str] = {}
    for g in results.values():
        for e in g.entities:
            c = canonical_name(e["name"]).lower()
            votes[c][e["label"]] += 1
            if len(e["name"]) > len(display.get(c, "")):
                display[c] = e["name"]
    priority = ["Person", "Location", "Organization", "Product", "Policy", "Event", "Concept"]
    final_label = {}
    for c, cnt in votes.items():
        if c in known_lc:
            final_label[c] = known_lc[c]
        else:
            top = max(cnt.values())
            final_label[c] = next(l for l in priority + list(cnt) if cnt.get(l) == top)

    # alias merge for one-word organisation names
    alias = {}
    orgs = [c for c, l in final_label.items() if l == "Organization"]
    for c in orgs:
        if " " not in c:
            longer = [o for o in orgs if o != c and o.split()[0] == c]
            if len(longer) == 1:
                alias[c] = longer[0]

    out = {}
    for doc_id, g in results.items():
        ng = ExtractedGraph(method=g.method)
        remap = {}
        for e in g.entities:
            c = canonical_name(e["name"]).lower()
            c2 = alias.get(c, c)
            key = ng.add_entity(final_label.get(c2, e["label"]), display.get(c2, e["name"]), e.get("description", ""))
            remap[e["key"]] = key
        for r in g.relations:
            if remap.get(r["src"]) and remap.get(r["dst"]):
                ng.add_relation(remap[r["src"]], remap[r["dst"]], r["type"], r.get("evidence", ""))
        out[doc_id] = ng
    return out


def load_documents_into_graph(store: GraphStore, docs: list[SourceDocument], known: dict[str, str],
                              progress: Progress = None) -> list[dict]:
    results: dict[str, ExtractedGraph] = {}
    for i, d in enumerate(docs):
        if progress:
            progress(f"Extracting entities: {d.title}", i / max(1, len(docs)))
        results[d.doc_id] = extract(d.text, known)
    results = resolve_entities(results, known)

    summaries = []
    for d in docs:
        g = results[d.doc_id]
        store.upsert_nodes("Document", [{
            "key": d.doc_id, "name": d.title, "source_type": d.source_type, "path": d.path,
            "chars": len(d.text), "content_hash": d.content_hash, "extraction_method": g.method,
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": d.text[:240].replace("\n", " ") + "…"}])
        by_label = defaultdict(list)
        for e in g.entities:
            by_label[e["label"]].append({"key": e["key"], "name": e["name"],
                                         **({"description": e["description"]} if e.get("description") else {})})
        for label, rows in by_label.items():
            store.upsert_nodes(label, rows)
        store.upsert_edges("MENTIONS", [{"src": d.doc_id, "dst": e["key"]} for e in g.entities])
        rel_rows = defaultdict(list)
        for r in g.relations:
            rel_rows[r["type"]].append({"src": r["src"], "dst": r["dst"],
                                        "props": {"evidence": r.get("evidence", ""), "source_doc": d.title}})
        for rel, rows in rel_rows.items():
            store.upsert_edges(rel, rows)
        if d.source_type == "email":
            frm, _ = email_people(d)
            for name in frm:
                key = schema.make_key("Person", canonical_name(name))
                store.upsert_nodes("Person", [{"key": key, "name": name}])
                store.upsert_edges("AUTHORED", [{"src": key, "dst": d.doc_id}])
        summaries.append({"doc_id": d.doc_id, "title": d.title, "type": d.source_type,
                          "entities": len(g.entities), "relations": len(g.relations), "method": g.method})
    return summaries


# ═════════════════════════════════════════════════════════════════════════════
# RAG corpus
# ═════════════════════════════════════════════════════════════════════════════
def document_chunks(docs: list[SourceDocument]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    out = []
    for d in docs:
        for j, piece in enumerate(splitter.split_text(d.text)):
            out.append({"id": f"{d.doc_id}#{j}", "text": piece, "title": d.title,
                        "source_type": d.source_type, "ref": d.doc_id})
    return out


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def insight_chunks(tx: pd.DataFrame, cases: pd.DataFrame) -> list[dict]:
    """Pre-computed aggregate facts so the RAG layer can answer analytical questions."""
    docs = []
    n, f = len(tx), int(tx["is_fraud"].sum())
    base = _pct(f, n)

    def add(cid, title, text):
        docs.append({"id": f"insight:{cid}", "title": title, "text": text, "source_type": "insight", "ref": ""})

    add("overall", "Portfolio overview",
        f"The portfolio contains {n:,} transactions, of which {f} are fraud ({base:.2f}% fraud rate). "
        f"Average amount is ${tx['amount'].mean():.2f}; average fraud amount is "
        f"${tx.loc[tx.is_fraud == 1, 'amount'].mean():.2f}. Total fraud value is "
        f"${tx.loc[tx.is_fraud == 1, 'amount'].sum():,.2f}.")

    def by(col, title, label):
        g = tx.groupby(col).agg(count=("is_fraud", "size"), fraud=("is_fraud", "sum"), avg=("amount", "mean"))
        g["rate"] = 100 * g["fraud"] / g["count"]
        g = g.sort_values("rate", ascending=False)
        parts = [f"{idx}: {int(r['count'])} transactions, {int(r['fraud'])} fraud ({r['rate']:.2f}%), "
                 f"avg amount ${r['avg']:.2f}" for idx, r in g.iterrows()]
        add(col, title, f"Fraud by {label}, highest fraud rate first. " + "; ".join(parts) +
            f". Highest: {g.index[0]} ({g['rate'].iloc[0]:.2f}%). Lowest: {g.index[-1]} ({g['rate'].iloc[-1]:.2f}%).")

    by("merchant_category", "Fraud by merchant category", "merchant category")
    by("time_slot", "Fraud by time of day (night vs daytime)", "time of day (time slot)")
    by("customer_segment", "Fraud by customer age segment", "customer age segment")

    parts = []
    for rf in schema.RISK_FACTORS:
        mask = tx["risk_factors"].map(lambda l: rf in l)
        c, fr = int(mask.sum()), int(tx.loc[mask, "is_fraud"].sum())
        rate = _pct(fr, c)
        parts.append(f"{rf}: {c} transactions, {fr} fraud ({rate:.2f}%, {rate / base if base else 0:.1f}x the "
                     f"{base:.2f}% baseline)")
    none = tx["risk_factor_count"] == 0
    parts.append(f"No risk factor: {int(none.sum())} transactions, {int(tx.loc[none, 'is_fraud'].sum())} fraud "
                 f"({_pct(tx.loc[none, 'is_fraud'].sum(), none.sum()):.2f}%)")
    add("risk_factors", "Fraud by risk factor", "Fraud rate and lift by risk factor. " + "; ".join(parts) + ".")

    fr = tx[tx.is_fraud == 1]
    combos = Counter(" + ".join(sorted(l)) if l else "no risk factor" for l in fr["risk_factors"]).most_common(6)
    add("combos", "Most common risk-factor combinations in fraud",
        "Among the fraud transactions, the most common risk-factor combinations are: " +
        "; ".join(f"{c} ({k} cases)" for c, k in combos) + ".")

    top = fr.sort_values("amount", ascending=False).head(10)
    add("top_fraud", "Largest fraud transactions by amount (biggest fraud losses)",
        "The 10 largest fraud transactions are: " + "; ".join(
            f"{r.tx_key} ${r.amount:.2f} {r.merchant_category} at {int(r.transaction_hour):02d}:00"
            for r in top.itertuples()) + ".")

    if not cases.empty:
        st = cases["status"].value_counts()
        an = cases["analyst"].value_counts()
        add("cases", "Fraud case management summary",
            f"There are {len(cases)} fraud cases in the case-management database. Status: " +
            ", ".join(f"{k}: {v}" for k, v in st.items()) + ". Cases per analyst: " +
            ", ".join(f"{k}: {v}" for k, v in an.items()) + ".")
    return docs


def build_corpus(tx: pd.DataFrame, cases: pd.DataFrame, docs: list[SourceDocument]) -> list[dict]:
    chunks = insight_chunks(tx, cases) + document_chunks(docs)
    for c in cases.to_dict("records") if not cases.empty else []:
        chunks.append({"id": f"case:{c['case_id']}", "title": c["case_id"], "source_type": "case",
                       "ref": c["tx_key"],
                       "text": f"Fraud case {c['case_id']} ({c['status']}, opened {c['opened_on']}) for "
                               f"transaction {c['tx_key']}, analyst {c['analyst']}. {c['notes']}"})
    for r in tx.to_dict("records"):
        chunks.append({"id": f"tx:{r['tx_key']}", "title": r["tx_key"], "source_type": "transaction",
                       "ref": r["tx_key"], "text": transaction_to_text(r)})
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# Entry points
# ═════════════════════════════════════════════════════════════════════════════
def all_source_documents() -> list[SourceDocument]:
    s = get_settings()
    docs = load_documents(s.sources_dir) + load_documents(s.storage_dir / "uploads")
    unique = {}
    for d in docs:
        unique.setdefault(d.content_hash, d)
    return list(unique.values())


def run_full_build(progress: Progress = None, clear: bool = True, store: Optional[GraphStore] = None) -> BuildReport:
    s = get_settings()
    rep = BuildReport(kind="full")
    t_all = time.time()

    def p(msg, frac):
        if progress:
            progress(msg, frac)
        print(f"[pipeline] {msg}")

    # 1. Extract + transform structured sources
    t0 = time.time(); p("Loading transactions and case database", 0.02)
    tx, q1 = load_transactions(s.transactions_csv)
    analysts, cases, q2 = load_case_database(s.cases_db)
    rep.quality = [q1.as_dict(), q2.as_dict()]
    rep.step("ingest_structured", t0, f"{len(tx):,} transactions, {len(cases)} cases")

    t0 = time.time(); p("Reading documents and emails", 0.08)
    docs = all_source_documents()
    rep.step("ingest_unstructured", t0, f"{len(docs)} documents/emails")

    # 2. Load graph
    store = store or connect()
    rep.backend = store.backend
    if clear:
        t0 = time.time(); p("Clearing existing graph", 0.1)
        store.clear()
        rep.step("clear_graph", t0)
    t0 = time.time(); p(f"Writing structured graph to {store.backend}", 0.15)
    counts = build_structured_graph(store, tx, analysts, cases)
    rep.step("graph_structured", t0, ", ".join(f"{k}={v}" for k, v in counts.items()))

    t0 = time.time()
    known = known_names(tx, analysts, docs)
    rep.documents = load_documents_into_graph(
        store, docs, known, progress=lambda m, f: p(m, 0.35 + 0.3 * f))
    methods = Counter(d["method"] for d in rep.documents)
    rep.step("graph_documents", t0, f"extraction methods: {dict(methods)}")
    store.save()
    rep.graph = store.stats()

    # 3. Vector index
    t0 = time.time(); p("Embedding corpus and building FAISS index", 0.7)
    index = VectorIndex()
    index.build(build_corpus(tx, cases, docs))
    rep.index = index.meta
    rep.step("vector_index", t0, f"{index.meta['chunks']:,} chunks via {index.meta['embedding_backend']}")

    rep.seconds = round(time.time() - t_all, 1)
    p("Done", 1.0)
    _log_run(rep)
    return rep


def ingest_uploaded_files(files: list[tuple[str, bytes]], store: Optional[GraphStore] = None,
                          index: Optional[VectorIndex] = None, progress: Progress = None) -> BuildReport:
    """Incremental update: new documents are added to the existing graph + index."""
    s = get_settings()
    rep = BuildReport(kind="incremental")
    t_all = time.time()
    up = s.storage_dir / "uploads"
    up.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    docs = []
    for name, data in files:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        try:
            d = parse_file(safe, data, path=f"storage/uploads/{safe}")
            if not d.text.strip():
                raise ValueError("no extractable text")
            (up / safe).write_bytes(data)
            docs.append(d)
        except Exception as exc:
            rep.errors.append(f"{name}: {exc}")
    rep.step("parse_uploads", t0, f"{len(docs)} parsed")
    if not docs:
        rep.seconds = round(time.time() - t_all, 1)
        return rep

    store = store or connect()
    rep.backend = store.backend
    tx, _ = load_transactions(s.transactions_csv)
    analysts, _, _ = load_case_database(s.cases_db)
    known = known_names(tx, analysts, all_source_documents())
    # include entity names already in the graph so new docs link to them
    for n in store.all_names(["Person", "Organization", "Location", "Product", "Policy"]):
        known.setdefault(n["name"], n["label"])

    t0 = time.time()
    rep.documents = load_documents_into_graph(store, docs, known, progress=progress)
    store.save()
    rep.step("graph_documents", t0)

    t0 = time.time()
    index = index or VectorIndex()
    index.add(document_chunks(docs))
    rep.step("vector_index", t0, f"index now {len(index.chunks):,} chunks")
    rep.graph = store.stats()
    rep.index = index.meta
    rep.seconds = round(time.time() - t_all, 1)
    _log_run(rep)
    return rep
