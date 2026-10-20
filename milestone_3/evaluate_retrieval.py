"""
evaluate_retrieval.py — Milestone 3: measure retrieval quality.

A small labelled set of questions with the source(s) that SHOULD be retrieved.
Reports Hit@k (was a correct source in the top k?) and MRR (how high was it?).

    python milestone_3/evaluate_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg.rag import VectorIndex  # noqa: E402

# (question, list of acceptable chunk-id prefixes or titles)
EVAL_SET = [
    ("Which merchant category has the highest fraud rate?", ["insight:merchant_category", "Q3 Fraud Risk Review"]),
    ("Is night-time riskier than daytime?", ["insight:time_slot", "Q3 Fraud Risk Review"]),
    ("How much riskier are foreign transactions?", ["insight:risk_factors", "Q3 Fraud Risk Review"]),
    ("Who approved the Sentinel DeviceID contract?", ["Vendor Agreement Sentinel DeviceID"]),
    ("Which company supplies the device trust score?", ["Vendor Agreement Sentinel DeviceID"]),
    ("What is velocity rule VR-5?", ["Policy VR-5 Velocity Limit"]),
    ("Describe the card-testing attack on travel merchants", ["Incident Report INC-2041"]),
    ("What did the Grocery deep-dive find?", ["RE: Night-time fraud spike - Grocery findings"]),
    ("When is Sentinel being rolled out in London?", ["Sentinel DeviceID rollout to London", "Vendor Agreement"]),
    ("How many fraud cases are still open?", ["insight:cases"]),
    ("What are the largest fraud transactions?", ["insight:top_fraud"]),
    ("Does customer age affect fraud?", ["insight:customer_segment", "Q3 Fraud Risk Review"]),
]


def is_hit(chunk: dict, targets: list[str]) -> bool:
    return any(chunk["id"].startswith(t) or chunk["title"].startswith(t) for t in targets)


def main(k: int = 5):
    index = VectorIndex()
    if not index.ready:
        print("Index not built. Run:  python milestone_2/main_m2.py   (or open the app once)")
        return
    hits, rr = 0, 0.0
    print(f"Embeddings: {index.meta.get('embedding_backend')} | chunks: {len(index.chunks):,}\n")
    for q, targets in EVAL_SET:
        res = index.search(q, k=k)
        rank = next((i for i, c in enumerate(res, 1) if is_hit(c, targets)), None)
        hits += rank is not None
        rr += 1 / rank if rank else 0
        print(f"{'HIT ' if rank else 'MISS'} rank={rank or '-':<2}  {q}")
    n = len(EVAL_SET)
    print(f"\nHit@{k}: {hits}/{n} = {hits / n:.0%}   MRR: {rr / n:.2f}")


if __name__ == "__main__":
    main()
