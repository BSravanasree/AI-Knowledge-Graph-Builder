"""
rag.py — RAG + Semantic Search Layer (Milestone 3).

  Embeddings : FastEmbed (BAAI/bge-small-en-v1.5, ONNX — no PyTorch needed).
               Falls back to TF-IDF + SVD if the model can't be downloaded.
  Vector DB  : FAISS (inner-product on L2-normalised vectors = cosine similarity),
               persisted to storage/rag/ and appended to incrementally.
  Retrieval  : vector search + source diversity (so 10k transaction rows don't
               crowd out documents) + knowledge-graph facts about entities
               mentioned in the question  (=> Graph-RAG).
  Generation : Gemini, instructed to answer only from the context and cite it.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import get_settings


# ═════════════════════════════════════════════════════════════════════════════
# Embeddings
# ═════════════════════════════════════════════════════════════════════════════
class Embedder:
    def __init__(self, model_name: str, store_dir: Path, force_tfidf: bool = False):
        self.model_name = model_name
        self.store_dir = store_dir
        self._fe = None
        self._tfidf = None
        self.backend = "tfidf"
        if not force_tfidf and get_settings().embedding_backend.lower() != "tfidf":
            try:
                from fastembed import TextEmbedding

                self._fe = TextEmbedding(model_name=model_name,
                                         cache_dir=str(get_settings().storage_dir / "models"))
                self.backend = f"fastembed:{model_name}"
            except Exception as exc:
                print(f"[rag] FastEmbed unavailable ({exc}); using TF-IDF embeddings.")
        if self._fe is None:
            p = store_dir / "tfidf.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    self._tfidf = pickle.load(f)

    def fit(self, texts: list[str]):
        """Only needed for the TF-IDF fallback (learns vocabulary + SVD)."""
        if self._fe is not None:
            return
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        dims = max(2, min(256, len(texts) - 1))
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                             max_features=50_000),
                             TruncatedSVD(n_components=dims, random_state=0))
        pipe.fit(texts)
        self._tfidf = pipe
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with open(self.store_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(pipe, f)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype="float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1
        return v / n

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self._fe is not None:
            return self._norm(np.array(list(self._fe.passage_embed(texts, batch_size=256))))
        return self._norm(self._tfidf.transform(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if self._fe is not None:
            return self._norm(np.array(list(self._fe.query_embed(text))))
        return self._norm(self._tfidf.transform([text]))


def _index_text(c: dict) -> str:
    return f"{c.get('title', '')}: {c['text']}"


# ═════════════════════════════════════════════════════════════════════════════
# Vector index (dense FAISS + sparse keyword index, fused => hybrid search)
# ═════════════════════════════════════════════════════════════════════════════
class VectorIndex:
    def __init__(self, directory: Optional[Path] = None):
        self.dir = directory or (get_settings().storage_dir / "rag")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[dict] = []
        self.meta: dict = {}
        self.index = None
        self.embedder: Optional[Embedder] = None
        self._sparse = None  # (TfidfVectorizer, csr_matrix) for keyword matching
        self._load()

    # persistence ─────────────────────────────────────────────────────────────
    def _load(self):
        import faiss

        ip, cp, mp = self.dir / "index.faiss", self.dir / "chunks.json", self.dir / "meta.json"
        if ip.exists() and cp.exists() and mp.exists():
            self.index = faiss.read_index(str(ip))
            self.chunks = json.loads(cp.read_text(encoding="utf-8"))
            self.meta = json.loads(mp.read_text(encoding="utf-8"))
            self.embedder = Embedder(self.meta.get("model", get_settings().embedding_model), self.dir,
                                     force_tfidf=self.meta.get("backend") == "tfidf")
            sp = self.dir / "sparse.pkl"
            if sp.exists():
                with open(sp, "rb") as f:
                    self._sparse = pickle.load(f)

    def _save(self):
        import faiss

        faiss.write_index(self.index, str(self.dir / "index.faiss"))
        (self.dir / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        if self._sparse is not None:
            with open(self.dir / "sparse.pkl", "wb") as f:
                pickle.dump(self._sparse, f)

    def _fit_sparse(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self._sparse = (vec, vec.fit_transform(texts))

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    # building ────────────────────────────────────────────────────────────────
    def build(self, chunks: list[dict]):
        import faiss

        s = get_settings()
        self.embedder = Embedder(s.embedding_model, self.dir)
        texts = [_index_text(c) for c in chunks]
        self.embedder.fit(texts)
        self._fit_sparse(texts)
        t0 = time.time()
        vecs = self.embedder.embed_passages(texts)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks = chunks
        self.meta = {"backend": "tfidf" if self.embedder.backend == "tfidf" else "fastembed",
                     "model": s.embedding_model, "embedding_backend": self.embedder.backend,
                     "dimension": int(vecs.shape[1]), "chunks": len(chunks),
                     "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "embed_seconds": round(time.time() - t0, 1)}
        self._save()

    def add(self, chunks: list[dict]):
        """Incremental update (new uploads) — no full rebuild."""
        if not self.ready:
            return self.build(chunks)
        existing = {c["id"] for c in self.chunks}
        chunks = [c for c in chunks if c["id"] not in existing]
        if not chunks:
            return
        texts = [_index_text(c) for c in chunks]
        vecs = self.embedder.embed_passages(texts)
        self.index.add(vecs)
        if self._sparse is not None:
            from scipy.sparse import vstack

            vec, mat = self._sparse
            self._sparse = (vec, vstack([mat, vec.transform(texts)]).tocsr())
        self.chunks.extend(chunks)
        self.meta["chunks"] = len(self.chunks)
        self._save()

    # search ──────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 6, max_per_type: Optional[dict] = None,
               source_types: Optional[list[str]] = None) -> list[dict]:
        if not self.ready:
            return []
        max_per_type = max_per_type if max_per_type is not None else {"transaction": 3, "case": 3}
        pool = min(len(self.chunks), max(80, k * 12))
        q = self.embedder.embed_query(query)
        d_scores, d_ids = self.index.search(q, pool)
        dense = {int(i): float(sc) for sc, i in zip(d_scores[0], d_ids[0]) if i >= 0}
        # keyword side: TF-IDF cosine
        sparse = {}
        if self._sparse is not None:
            vec, mat = self._sparse
            sims = (mat @ vec.transform([query]).T).toarray().ravel()
            top = np.argsort(-sims)[:pool]
            sparse = {int(i): float(sims[i]) for i in top if sims[i] > 0}
        # reciprocal-rank fusion (robust to the two score scales)
        fused: dict[int, float] = {}
        for ranking in (sorted(dense, key=lambda i: -dense[i]), sorted(sparse, key=lambda i: -sparse[i])):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        out, per_type = [], {}
        for i in sorted(fused, key=lambda i: -fused[i]):
            score = dense.get(i, 0.0)
            c = self.chunks[i]
            st = c.get("source_type", "")
            if source_types and st not in source_types:
                continue
            if per_type.get(st, 0) >= max_per_type.get(st, k):
                continue
            per_type[st] = per_type.get(st, 0) + 1
            out.append({**c, "score": round(fused[i] * 60, 3), "dense": round(score, 3),
                        "keyword": round(sparse.get(i, 0.0), 3)})
            if len(out) >= k:
                break
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Graph-RAG answer
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    used_llm: bool = False
    model: str = ""
    seconds: float = 0.0


SYSTEM_PROMPT = (
    "You are an enterprise intelligence analyst. Answer the user's question using ONLY the "
    "numbered context passages and knowledge-graph facts provided. Cite passages like [1] and "
    "graph facts like [G2]. Be concise and specific (numbers, names). If the context does not "
    "contain the answer, say so plainly — do not guess."
)


def find_entities_in_question(question: str, names: list[dict]) -> list[dict]:
    """Match graph node names (and TX_/CASE- ids) that appear in the question."""
    q = question.lower()
    hits = []
    for n in sorted(names, key=lambda x: -len(str(x["name"]))):
        if n.get("label") in ("FraudStatus",):
            continue  # "fraud" appears in nearly every question
        name = str(n["name"]).lower()
        if len(name) < 3:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", q) and not any(name in h["name"].lower() for h in hits):
            hits.append(n)
    for tx in re.findall(r"\btx[_ ]?(\d+)\b", q):
        hits.append({"key": f"TX_{tx}", "name": f"TX_{tx}", "label": "Transaction"})
    return hits[:8]


def answer_question(question: str, index: VectorIndex, store, names: list[dict],
                    history: Optional[list[dict]] = None, k: Optional[int] = None) -> Answer:
    t0 = time.time()
    s = get_settings()
    k = k or s.top_k
    sources = index.search(question, k=k)
    ents = find_entities_in_question(question, names)
    facts = store.facts_text([e["key"] for e in ents], per_node=10) if ents else []
    facts = facts[:30]

    ctx = "\n".join(f"[{i}] ({c['source_type']}: {c['title']}) {c['text']}" for i, c in enumerate(sources, 1))
    gctx = "\n".join(f"[G{i}] {f}" for i, f in enumerate(facts, 1)) or "(none)"
    hist = ""
    if history:
        hist = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in history[-4:])
        hist = f"Conversation so far:\n{hist}\n\n"
    prompt = (f"{hist}Context passages:\n{ctx or '(none)'}\n\nKnowledge-graph facts:\n{gctx}\n\n"
              f"Question: {question}\nAnswer:")

    if s.llm_enabled:
        try:
            from .llm import active_model, generate

            text = generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            return Answer(question, text, sources, facts, [e["name"] for e in ents], True,
                          active_model(), round(time.time() - t0, 2))
        except Exception as exc:
            note = f"_LLM call failed ({exc}). Showing retrieved evidence instead._\n\n"
    else:
        note = "_No GEMINI_API_KEY configured — showing the most relevant evidence (retrieval only)._\n\n"

    def snippet(t):
        t = re.sub(r"\s+", " ", t).strip()
        return t if len(t) <= 320 else t[:317] + "…"

    bullets = [f"- **[{i}] {c['title']}** — {snippet(c['text'])}" for i, c in enumerate(sources[:4], 1)]
    if facts:
        bullets.append("\n**Graph facts**")
        bullets += [f"- {f}" for f in facts[:8]]
    return Answer(question, note + "\n".join(bullets), sources, facts, [e["name"] for e in ents],
                  False, "", round(time.time() - t0, 2))
