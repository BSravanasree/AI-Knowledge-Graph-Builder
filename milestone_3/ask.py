"""
Milestone 3 — ask the knowledge base from the terminal.

    python milestone_3/ask.py "Which merchant category has the highest fraud rate?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg.graph_store import connect  # noqa: E402
from kg.rag import VectorIndex, answer_question  # noqa: E402
from dashboard.state import ENTITY_NAME_LABELS  # noqa: E402

q = " ".join(sys.argv[1:]) or "Which merchant category has the highest fraud rate?"
store, index = connect(), VectorIndex()
ans = answer_question(q, index, store, store.all_names(ENTITY_NAME_LABELS))
print(f"Q: {q}\n\n{ans.answer}\n")
print("Sources:")
for i, c in enumerate(ans.sources, 1):
    print(f"  [{i}] {c['source_type']:<11} {c['title']}")
for i, f in enumerate(ans.graph_facts[:10], 1):
    print(f"  [G{i}] {f}")
