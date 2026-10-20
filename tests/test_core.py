"""Run:  pytest -q        (uses a temporary storage folder and the local graph)"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    yield


def test_relation_normalisation_is_injection_safe():
    from kg.schema import normalise_relation

    assert normalise_relation("partnered with") == "PARTNERED_WITH"
    assert normalise_relation("INJECTED; DROP DATABASE") == "RELATED_TO"
    assert normalise_relation("`) DETACH DELETE n //") == "RELATED_TO"


def test_every_source_type_parses():
    from kg.config import get_settings
    from kg.ingestion import load_documents

    docs = load_documents(get_settings().sources_dir)
    kinds = {d.source_type for d in docs}
    assert {"pdf", "docx", "markdown", "text", "email"} <= kinds
    assert all(len(d.text) > 200 for d in docs)


def test_transaction_validation_rejects_bad_rows(tmp_path):
    import pandas as pd

    from kg.ingestion import load_transactions

    df = pd.read_csv(ROOT / "data" / "creditcard.csv").head(20)
    df.loc[0, "amount"] = -5
    df.loc[1, "transaction_hour"] = 30
    df = pd.concat([df, df.head(1)])  # duplicate
    p = tmp_path / "tx.csv"
    df.to_csv(p, index=False)
    out, rep = load_transactions(p)
    assert rep.duplicates_removed == 1
    assert len(out) == 18 and rep.rule_failures


def test_llm_extraction_path(monkeypatch):
    """The LLM branch, with Gemini mocked (no network)."""
    from kg import extraction

    def fake_generate(prompt, schema=None, **_):
        return schema(entities=[{"name": "Acme Ltd", "type": "company"},
                                {"name": "Pune", "type": "city"}],
                      relations=[{"source": "Acme Ltd", "relation": "located in", "target": "Pune"}])

    monkeypatch.setattr("kg.llm.generate", fake_generate)
    g = extraction.extract_llm("Acme Ltd is based in Pune.", known={})
    assert {e["label"] for e in g.entities} == {"Organization", "Location"}
    assert g.relations[0]["type"] == "LOCATED_IN"


def test_full_build_and_graph_rag():
    from kg.graph_store import LocalStore
    from kg.pipeline import run_full_build
    from kg.rag import VectorIndex, answer_question

    rep = run_full_build(store=LocalStore())
    assert rep.graph["labels"]["Transaction"] == 10000
    assert rep.graph["rel_types"]["INVESTIGATES"] >= 151
    store, index = LocalStore(), VectorIndex()
    nodes, edges = store.subgraph(search="grocery", max_nodes=60)
    assert nodes and edges and len(nodes) <= 60
    names = store.all_names(["MerchantCategory", "Person", "Organization", "RiskFactor"])
    ans = answer_question("What is the fraud rate for Grocery?", index, store, names)
    assert ans.sources and any("Grocery" in f for f in ans.graph_facts)


def test_incremental_upload():
    from kg.graph_store import LocalStore
    from kg.pipeline import ingest_uploaded_files, run_full_build
    from kg.rag import VectorIndex

    run_full_build(store=LocalStore())
    store, index = LocalStore(), VectorIndex()
    before = store.stats()["labels"].get("Document", 0)
    rep = ingest_uploaded_files([("memo.txt", b"Harbor Trust Bank partnered with Orion Pay in Mumbai to "
                                               b"reduce Travel fraud. Priya Raman approved the pilot.")],
                                store=store, index=index)
    assert not rep.errors
    assert store.stats()["labels"]["Document"] == before + 1
    assert any(c["title"] == "memo" for c in index.chunks)
