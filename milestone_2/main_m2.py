"""
Milestone 2 — build the knowledge graph (and the vector index used by M3).

    python milestone_2/main_m2.py            # full rebuild (clears the graph first)
    python milestone_2/main_m2.py --no-clear # merge into the existing graph

Uses Neo4j when NEO4J_URI/NEO4J_PASSWORD are set and reachable, otherwise the
local NetworkX graph in storage/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg.pipeline import run_full_build  # noqa: E402

if __name__ == "__main__":
    rep = run_full_build(clear="--no-clear" not in sys.argv)
    print(f"\nBackend: {rep.backend} | {rep.graph['nodes']:,} nodes | {rep.graph['relationships']:,} relationships "
          f"| {rep.seconds}s")
    for step in rep.steps:
        print(f"  {step['step']:<20} {step['seconds']:>6}s  {step['detail']}")
    print("\nDocuments:")
    for d in rep.documents:
        print(f"  {d['title'][:50]:<50} entities={d['entities']:<3} relations={d['relations']:<3} via {d['method']}")
