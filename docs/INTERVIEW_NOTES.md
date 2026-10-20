# Interview notes: how to explain this project

## 30-second pitch
"I built an AI platform that turns scattered enterprise data into a knowledge graph. It takes a transactions table, a case-management database, PDFs, Word documents and emails. It cleans and validates them, extracts entities and relationships with an LLM, and resolves them onto shared nodes in Neo4j. On top of that sits a Graph-RAG layer: hybrid vector + keyword search in FAISS, plus facts pulled from the graph, answered by Gemini with citations. A Streamlit dashboard lets users explore the graph, ask questions and upload new documents, and those updates go into the live graph without a rebuild. It's deployed on Streamlit Cloud with Neo4j AuraDB."

## Design decisions you should be able to defend
| Decision | Why |
|---|---|
| Typed schema (Transaction, MerchantCategory, RiskFactor, Person…) instead of one generic `:Entity` label | Enables meaningful Cypher queries and filtering, and keeps the graph explainable |
| Shared dimension nodes (merchant, time slot, risk factor) | The dataset has no cardholder ID, so linking every transaction to its *own* user node would create 10k disconnected pairs. Shared nodes create real paths between transactions |
| LLM extraction with a **response schema** (pydantic) | The model must return valid JSON, so there's no fragile parsing. Entity types and relation names are restricted to the schema vocabulary |
| spaCy fallback + content-hash cache | The app works without an API key, and rebuilds don't re-spend LLM quota |
| Entity resolution (gazetteer + label voting + alias merge) | "Sentinel", "Sentinel DeviceID" and "Sentinel DeviceID Ltd" become one node, and document mentions link to structured nodes |
| `UNWIND … MERGE` in batches of 1000, uniqueness constraint on `key` | About 45 round trips instead of 45k. Runs are idempotent, so re-running the pipeline never duplicates data |
| Relationship types validated to `[A-Z0-9_]` | Cypher can't parameterise relationship types, so validation is what prevents Cypher injection |
| Hybrid retrieval (dense + TF-IDF, reciprocal-rank fusion) | Embeddings capture meaning, keywords catch exact IDs and names (TX_2051, VR-5) |
| Pre-computed "insight" chunks | Vector search can't aggregate over 10k rows, so aggregate statistics are indexed as text |
| Graph facts in the prompt (Graph-RAG) | Answers combine unstructured text with structured counts and relationships |
| FastEmbed (ONNX) instead of sentence-transformers | No PyTorch dependency, so a much smaller install. That matters on free hosting |
| Local NetworkX fallback | The demo never breaks if the database is paused |

## Questions to expect
* *How would you scale to millions of documents?* Batch and stream ingestion (Kafka or Databricks Auto Loader), extraction as a parallel job, a managed vector DB (Pinecone or Databricks Vector Search), Neo4j with `CALL {…} IN TRANSACTIONS`, and incremental runs keyed by content hash. All of these already exist here in small form.
* *How do you measure quality?* `milestone_3/evaluate_retrieval.py` reports Hit@k and MRR on a labelled question set. For extraction you would spot-check precision against hand-labelled triples. For answers, use groundedness checks; the prompt already forces "say so if it's not in the context".
* *What about LLM hallucination?* Context-only prompting, citations, temperature 0 for extraction, a schema-constrained output vocabulary, and evidence quotes stored on every extracted relationship.
* *Why a graph and not just SQL?* Multi-hop questions ("which analysts investigate fraud that shares risk factors with the incident Arjun reported?") are natural path queries in a graph, and adding new entity types needs no schema migration.
* *Security?* Secrets live in env vars or Streamlit secrets and are never committed, Cypher is parameterised, relationship types use an allowlist, and an optional app password protects the deployed app.

## Demo script (3 minutes)
1. **Overview:** show the architecture and counts.
2. **Graph explorer:** search "Grocery". Show emails and the Q3 report linking to the same Grocery node that the transactions use, then drill down to its fraud rate.
3. **Ask:** "Who approved the Sentinel DeviceID agreement?" then "Which risk factors have the biggest lift?" Open the Evidence panel.
4. **Data sources:** upload a new memo, then search for it in the explorer. This shows the incremental update.
