# AI Knowledge Graph Builder for Enterprise Intelligence

An AI platform that builds a **knowledge graph** from several enterprise sources: a transactions table, a case-management database, PDF/Word/Markdown/text documents and emails. It then answers questions over that graph with **Graph-RAG** (semantic search plus graph facts plus an LLM), and an **interactive dashboard** lets you explore the graph.

> Demo domain: card-fraud intelligence for a fictional card issuer, "Meridian Card Services". All people, companies, documents and case records in `data/` are **synthetic**. The transaction statistics quoted in the documents match the real patterns in `data/creditcard.csv`.

| Module (from the brief) | Where it lives | What it does |
|---|---|---|
| Data Ingestion & Processing | `kg/ingestion.py` | Reads CSV/DB tables, PDF, DOCX, MD, TXT and `.eml`. Cleans, de-duplicates, applies validation rules (bad rows are rejected and counted) and adds derived fields |
| Entity & Relationship Extraction | `kg/extraction.py` | **Gemini LLM** with schema-forced JSON output, a **spaCy** NER + rule fallback, gazetteer linking and cross-document entity resolution |
| Graph Construction & Storage | `kg/graph_store.py`, `kg/schema.py` | Typed property graph in **Neo4j** (batched `UNWIND … MERGE`, uniqueness constraint, idempotent and incremental). Falls back to a local NetworkX graph |
| RAG + Semantic Search | `kg/rag.py` | **FastEmbed** (bge-small, ONNX) + **FAISS**, hybrid dense + keyword retrieval (reciprocal-rank fusion), graph facts, cited Gemini answers. Chunking uses **LangChain** text splitters |
| Interactive Graph Dashboard | `app.py`, `dashboard/` | Streamlit app: graph explorer with filters, search and drill-down; Q&A chat; document upload; Plotly analytics |

```
Sources ──► M1 ingest (clean·validate·enrich) ──► M2 extract (LLM / spaCy) ──► Neo4j graph ─┐
   │                                                                                        ├─► M3 Graph-RAG ─► M4 dashboard
   └───────────────────────────────► chunk + embed ──────────────────────► FAISS index ─────┘
```

### Graph schema
```
(:Transaction)-[:AT_MERCHANT]->(:MerchantCategory)      (:Person)-[:INVESTIGATES {case_id,status}]->(:Transaction)
(:Transaction)-[:IN_TIMESLOT]->(:TimeSlot)              (:Person)-[:LOCATED_IN]->(:Location)
(:Transaction)-[:BY_SEGMENT]->(:CustomerSegment)        (:Person)-[:AUTHORED]->(:Document)
(:Transaction)-[:HAS_RISK_FACTOR]->(:RiskFactor)        (:Document)-[:MENTIONS]->(any entity)
(:Transaction)-[:FLAGGED_AS]->(:FraudStatus)            (:Organization)-[:PARTNERED_WITH|SUPPLIES|…]->(…)  ← extracted
```
Document entities resolve to the **same nodes** as the structured data. For example, the email that says "Grocery" links to the `Grocery` merchant node that 1,944 transactions point at. That cross-source link is the "hidden relationship" the brief asks for.

---

## 1 · Run it on your laptop (VS Code)

**Prerequisites:** Python **3.12** (or 3.11) from python.org, Git, and VS Code with the *Python* extension.
Python 3.14 is too new for spaCy and faiss wheels. If `python --version` shows 3.14, install 3.12 alongside it.

```powershell
# 1. Open the folder in VS Code → Terminal → New Terminal (PowerShell)
cd AI-Based-Knowledge-Graph-Builder-for-Enterprise-Intelligence

# 2. Create a fresh virtual environment with Python 3.12
#    (delete any old .venv first: Remove-Item -Recurse -Force .venv)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If PowerShell blocks the script once, run:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install dependencies (about 3–5 minutes the first time)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure keys (optional — the app also runs without them)
copy .env.example .env
#    Edit .env: GEMINI_API_KEY=... (free at https://aistudio.google.com/apikey)
#    Neo4j: NEO4J_URI / NEO4J_PASSWORD (see below), or leave blank for the local graph

# 5. Check the setup
python scripts/check_setup.py

# 6. Launch
streamlit run app.py
```
macOS/Linux: use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

The browser opens at <http://localhost:8501>. On first launch the app builds the knowledge base automatically (about 1–2 minutes; the embedding model downloads once). In VS Code, choose the interpreter with **Ctrl+Shift+P → Python: Select Interpreter → .venv**.

**Using Neo4j locally (optional).** Install Neo4j Desktop, create a database, start it, then set the following in `.env`:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<the password you chose>
```
The sidebar shows **Graph: Neo4j** when it's connected. Open Neo4j Browser and try:
```cypher
MATCH (p:Person)-[:INVESTIGATES]->(t:Transaction)-[:HAS_RISK_FACTOR]->(r:RiskFactor)
RETURN p.name, r.name, count(*) AS cases ORDER BY cases DESC
```

### Milestone scripts (command line)
| | Command |
|---|---|
| M1 | `python milestone_1/run_ingestion.py` prints the quality report and schema, and writes `data/processed_creditcard.csv`. The notebook is in `milestone_1/` |
| M2 | `python milestone_2/main_m2.py` builds the graph and index (`--no-clear` merges instead) |
| M3 | `python milestone_3/ask.py "your question"` · `python milestone_3/evaluate_retrieval.py` (Hit@5 / MRR) |
| M4 | `streamlit run app.py` |
| Tests | `pip install pytest && pytest -q` |

---

## 2 · Make it live (free)

**Recommended setup:** Streamlit Community Cloud for the app, Neo4j AuraDB Free for the graph and Gemini for the LLM.

1. **Push to GitHub.** `.env`, `storage/` and `.streamlit/secrets.toml` are git-ignored, so secrets don't get pushed.
   ```bash
   git add .
   git commit -m "Knowledge graph builder: multi-source ingestion, LLM extraction, Graph-RAG, dashboard"
   git push
   ```
2. **Create a Neo4j AuraDB Free instance** at <https://console.neo4j.io>. Download the credentials file; it contains the `neo4j+s://…` URI and the password. The free tier allows 200k nodes and 400k relationships, and this project uses about 10k and 45k.
3. **Deploy** at <https://share.streamlit.io>: **Create app**, then pick your repo, branch `main`, main file `app.py`. Under **Advanced settings**, choose Python **3.12** and paste the secrets:
   ```toml
   GEMINI_API_KEY = "..."
   NEO4J_URI = "neo4j+s://xxxx.databases.neo4j.io"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "..."
   # APP_PASSWORD = "optional — locks the app behind a password"
   ```
4. Click **Deploy**. The first visit builds the graph in Aura and the vector index, and you get a public `https://<name>.streamlit.app` URL for your resume.

Notes:
* Free Streamlit apps sleep when unused. If yours is asleep, open the URL a minute before an interview.
* Free Aura instances can be paused when idle. If the sidebar shows "local graph", resume the instance in the Aura console. The app keeps working on the local graph in the meantime.
* **Alternative host:** a `Dockerfile` is included. It works on Hugging Face Spaces (Docker, set port 8501), Render, Railway or Cloud Run. Add the same values as environment variables.

---

## 3 · Configuration
| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Enables LLM extraction and generated answers |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Any current Gemini Flash model. The app falls back through newer and older Flash names automatically |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | – | Graph database |
| `GRAPH_BACKEND` | `auto` | `auto` uses Neo4j when it's reachable, otherwise local; you can also force `neo4j` or `local` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `EMBEDDING_BACKEND` | `auto` | `tfidf` skips the model download (offline fallback) |
| `USE_LLM_EXTRACTION` | `true` | Set `false` to save LLM quota |
| `APP_PASSWORD` | – | Simple password gate for the deployed app |

## 4 · Project structure
```
app.py                  Streamlit entry point (multipage navigation)
kg/                     core library
  config.py             settings from env / .env / Streamlit secrets
  schema.py             graph schema, business rules, relation normalisation
  ingestion.py          M1 readers, cleaning, validation, enrichment
  extraction.py         M2 LLM + spaCy extraction, caching
  graph_store.py        M2 Neo4j + local graph back-ends
  pipeline.py           ETL orchestration, entity resolution, run log
  rag.py                M3 embeddings, FAISS, hybrid search, Graph-RAG answers
  llm.py                Gemini client with model fallback and retry
dashboard/              M4 pages (overview, sources, extraction, ask, explorer, insights)
milestone_1..4/         milestone runners + the M1 notebook
data/                   creditcard.csv, enterprise.db, sources/{documents,emails}
scripts/                check_setup.py, make_sample_sources.py
tests/                  pytest suite (runs offline)
storage/                generated: graph.pkl, FAISS index, caches, uploads (git-ignored)
```
