"""
check_setup.py — run this first if something doesn't work.

    python scripts/check_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ok = lambda m: print(f"  [OK]   {m}")      # noqa: E731
warn = lambda m: print(f"  [WARN] {m}")    # noqa: E731

print(f"Python {sys.version.split()[0]}")
if sys.version_info >= (3, 14):
    warn("Python 3.14 is too new for some packages — install Python 3.12 and recreate the venv.")

from kg.config import get_settings  # noqa: E402

s = get_settings()

try:
    import spacy

    spacy.load("en_core_web_sm")
    ok("spaCy + en_core_web_sm")
except Exception as e:
    warn(f"spaCy model missing ({e}). Run: python -m spacy download en_core_web_sm")

if s.neo4j_configured:
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as d:
            d.verify_connectivity()
        ok(f"Neo4j reachable at {s.neo4j_uri}")
    except Exception as e:
        warn(f"Neo4j not reachable ({e}). App will use the local graph.")
else:
    warn("NEO4J_URI / NEO4J_PASSWORD not set — app will use the local graph.")

if s.llm_enabled:
    try:
        from kg.llm import active_model, generate

        generate("Reply with the single word: ready")
        ok(f"Gemini API key works (model {active_model()})")
    except Exception as e:
        warn(f"Gemini call failed: {e}")
else:
    warn("GEMINI_API_KEY not set — spaCy extraction + retrieval-only answers.")

try:
    from fastembed import TextEmbedding

    TextEmbedding(model_name=s.embedding_model, cache_dir=str(s.storage_dir / "models"))
    ok(f"FastEmbed model {s.embedding_model} downloaded")
except Exception as e:
    warn(f"FastEmbed model unavailable ({e}) — TF-IDF fallback will be used.")
