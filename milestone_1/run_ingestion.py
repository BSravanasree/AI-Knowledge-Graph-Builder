"""
Milestone 1 — Data ingestion & schema.

    python milestone_1/run_ingestion.py

Loads every source (transactions CSV, case-management DB, documents, emails),
runs cleaning + validation, prints the data-quality report and the graph
schema, and writes data/processed_creditcard.csv (same output as the notebook).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg import schema  # noqa: E402
from kg.config import get_settings  # noqa: E402
from kg.ingestion import load_case_database, load_transactions  # noqa: E402
from kg.pipeline import all_source_documents  # noqa: E402

s = get_settings()
tx, q1 = load_transactions(s.transactions_csv)
analysts, cases, q2 = load_case_database(s.cases_db)
docs = all_source_documents()

print("== Data quality ==")
for q in (q1, q2):
    print(q.as_dict())
print(f"\n== Unstructured sources ({len(docs)}) ==")
for d in docs:
    print(f"  [{d.source_type:8}] {d.title}  ({len(d.text):,} chars)")
print("\n== Knowledge graph schema ==")
print("  Node labels   :", ", ".join(schema.ALL_LABELS))
print("  Structured rel:", ", ".join(schema.STRUCTURED_RELATIONS))
print("  Extracted rel :", ", ".join(schema.EXTRACTED_RELATIONS))

out = s.data_dir / "processed_creditcard.csv"
tx.assign(risk_factors=tx["risk_factors"].map("|".join)).to_csv(out, index=False)
print(f"\nWrote {out.relative_to(s.data_dir.parent)} ({len(tx):,} rows)")
