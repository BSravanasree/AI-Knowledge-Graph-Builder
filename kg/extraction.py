"""
extraction.py — Entity & Relationship Extraction Engine (Milestone 2).

Three layers, combined:

  1. LLM extraction (primary)   — Gemini reads each chunk and returns typed
     entities + (subject, relation, object) triples as JSON that is forced to
     match a schema (no fragile string parsing).
  2. spaCy extraction (fallback) — NER + dependency-parse SVO triples, used
     when no API key is configured or the LLM call fails.
  3. Gazetteer linking (always)  — known names from the structured data
     (merchant categories, risk factors) are matched in the text so documents
     link onto the SAME nodes the transactions use (cross-source entity
     resolution).

Results are cached on disk by content hash, so rebuilding the graph does not
re-spend LLM quota on documents that haven't changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field

from . import schema
from .config import get_settings

# ── LLM output schema ─────────────────────────────────────────────────────────
class _Entity(BaseModel):
    name: str = Field(description="Canonical name, e.g. 'Sentinel DeviceID Ltd'")
    type: str = Field(description="One of: " + ", ".join(schema.ENTITY_LABELS))
    description: str = Field(default="", description="Short description from the text")


class _Relation(BaseModel):
    source: str
    relation: str = Field(description="One of: " + ", ".join(schema.EXTRACTED_RELATIONS))
    target: str
    evidence: str = Field(default="", description="Short quote from the text supporting it")


class _Extraction(BaseModel):
    entities: list[_Entity]
    relations: list[_Relation]


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class ExtractedGraph:
    entities: list[dict] = field(default_factory=list)   # {key,label,name,description}
    relations: list[dict] = field(default_factory=list)  # {src,dst,type,evidence}
    method: str = ""

    def add_entity(self, label: str, name: str, description: str = "") -> Optional[str]:
        name = _tidy_name(name)
        if not _valid_name(name):
            return None
        key = schema.make_key(label, canonical_name(name))
        for e in self.entities:
            if e["key"] == key:
                if len(name) > len(e["name"]):
                    e["name"] = name
                if description and not e.get("description"):
                    e["description"] = description
                return key
        self.entities.append({"key": key, "label": label, "name": name, "description": description})
        return key

    def add_relation(self, src: str, dst: str, rel: str, evidence: str = ""):
        if not src or not dst or src == dst:
            return
        item = {"src": src, "dst": dst, "type": schema.normalise_relation(rel), "evidence": evidence[:300]}
        if not any(r["src"] == src and r["dst"] == dst and r["type"] == item["type"] for r in self.relations):
            self.relations.append(item)

    def key_for(self, name: str) -> Optional[str]:
        target = canonical_name(_tidy_name(name)).lower()
        for e in self.entities:
            if canonical_name(e["name"]).lower() == target:
                return e["key"]
        return None


_ORG_SUFFIX = re.compile(r"\b(ltd|limited|inc|llc|plc|pvt|corp|corporation|co)\.?$", re.I)


def canonical_name(name: str) -> str:
    """'Sentinel DeviceID Ltd.' and 'Sentinel DeviceID' resolve to the same node."""
    n = re.sub(r"\s+", " ", name).strip(" .,;:'\"")
    n = _ORG_SUFFIX.sub("", n).strip(" .,")
    n = re.sub(r"^the\s+", "", n, flags=re.I)
    return n


def _tidy_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip(" .,;:'\"()[]")


def _valid_name(name: str) -> bool:
    return 2 <= len(name) <= 80 and not name.isdigit() and re.search(r"[A-Za-z]", name) is not None


# ── Gazetteer: names that already exist in the structured graph ──────────────
def gazetteer_link(text: str, graph: ExtractedGraph, known: dict[str, str]):
    """`known` maps display name -> label (e.g. {'Grocery': 'MerchantCategory'})."""
    low = text.lower()
    for name, label in known.items():
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", low):
            graph.add_entity(label, name)


# ── LLM extraction ───────────────────────────────────────────────────────────
_PROMPT = """You are building an enterprise knowledge graph from internal documents.
Extract the important entities and the factual relationships between them.

Rules:
- Entity types allowed: {labels}.
- Use MerchantCategory for these merchant categories: {merchants}.
- Use RiskFactor for these fraud risk factors: {risks}.
- Relation types allowed: {relations}. Use RELATED_TO only if nothing else fits.
- Use full canonical names (e.g. "Priya Raman", not "Priya"). Do not invent facts.
- Every relation's source and target must also appear in the entities list.

Text:
\"\"\"
{text}
\"\"\"
"""


def extract_llm(text: str, known: dict[str, str]) -> ExtractedGraph:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .llm import generate

    splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=300)
    graph = ExtractedGraph(method="llm")
    merchants = [n for n, l in known.items() if l == "MerchantCategory"]
    risks = [n for n, l in known.items() if l == "RiskFactor"]
    for chunk in splitter.split_text(text):
        prompt = _PROMPT.format(
            labels=", ".join(schema.ENTITY_LABELS), relations=", ".join(schema.EXTRACTED_RELATIONS),
            merchants=", ".join(merchants) or "-", risks=", ".join(risks) or "-", text=chunk,
        )
        result: _Extraction = generate(prompt, schema=_Extraction, temperature=0.0)
        for ent in result.entities:
            graph.add_entity(schema.normalise_label(ent.type), ent.name, ent.description)
        for rel in result.relations:
            src, dst = graph.key_for(rel.source), graph.key_for(rel.target)
            if src is None:
                src = graph.add_entity("Concept", rel.source)
            if dst is None:
                dst = graph.add_entity("Concept", rel.target)
            graph.add_relation(src, dst, rel.relation, rel.evidence)
    return graph


# ── spaCy extraction (fallback) ──────────────────────────────────────────────
SPACY_LABELS = {
    "ORG": "Organization", "PERSON": "Person", "GPE": "Location", "LOC": "Location",
    "FAC": "Location", "PRODUCT": "Product", "EVENT": "Event", "LAW": "Policy",
    "NORP": None, "DATE": None, "TIME": None, "MONEY": None, "PERCENT": None,
    "CARDINAL": None, "ORDINAL": None, "QUANTITY": None,
}


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def spacy_available() -> bool:
    try:
        _nlp()
        return True
    except Exception:
        return False


_STOP_NAMES = {"email", "fraud", "ip", "subject", "date", "team", "thanks", "regards", "hi", "hello"}
_TITLE_WORDS = {"head", "chief", "lead", "fraud", "data", "information", "security", "officer",
                "analyst", "engineer", "manager", "director", "compliance", "risk"}
_GENERIC_WORDS = _TITLE_WORDS | {"team", "review", "control", "operations", "department", "q1", "q2",
                                 "q3", "q4", "report", "unit", "office"}


def _clean_spacy_name(name: str, label: str) -> str:
    name = re.sub(r"^[^A-Za-z]+", "", name).strip()
    name = re.sub(r"^the\s+", "", name, flags=re.I)
    if label == "Person":
        kept = []
        for tok in name.split():
            if kept and tok.lower() in _TITLE_WORDS:
                break
            kept.append(tok)
        name = " ".join(kept)
    return name


def extract_spacy(text: str, known: dict[str, str]) -> ExtractedGraph:
    """
    Fallback extractor: spaCy NER + verb-anchored relation extraction.

    Relations: in each sentence, a verb from schema.VERB_TO_RELATION links the
    nearest entity before it (subject) to the nearest entity after it (object);
    passive voice ("was approved by X") swaps the direction.
    """
    nlp = _nlp()
    graph = ExtractedGraph(method="spacy")
    doc = nlp(text)
    low = text.lower()

    spans: list[tuple[int, int, str]] = []  # (start_token, end_token, key)

    def overlaps(a0, a1):
        return any(not (a1 <= s0 or a0 >= s1) for s0, s1, _ in spans)

    # 1) known names first (gazetteer) — authoritative label and boundaries
    for name, label in sorted(known.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(r"\b" + re.escape(name.lower()) + r"\b", low):
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if span is None or overlaps(span.start, span.end):
                continue
            key = graph.add_entity(label, name)
            if key:
                spans.append((span.start, span.end, key))

    # 2) statistical NER for everything else
    for ent in doc.ents:
        label = SPACY_LABELS.get(ent.label_, "Concept")
        if label is None or overlaps(ent.start, ent.end):
            continue
        name = _clean_spacy_name(ent.text, label)
        if name.lower() in _STOP_NAMES or len(name) < 3:
            continue
        if all(tok.lower() in _GENERIC_WORDS for tok in name.split()):
            continue  # job titles / generic phrases like "Lead Data Engineer", "Fraud Team"
        if label == "Organization" and name.split()[-1].lower() in {"policy", "rule"}:
            label = "Policy"
        if label == "Person" and len(name.split()) < 2:
            # single first names ("Arjun") -> a known full name if unambiguous
            matches = [k for k, l in known.items() if l == "Person" and k.split()[0] == name]
            if len(matches) != 1:
                continue
            name = matches[0]
        key = graph.add_entity(label, name)
        if key:
            spans.append((ent.start, ent.end, key))

    spans.sort()

    # 3) relations
    for sent in doc.sents:
        sent_spans = [sp for sp in spans if sp[0] >= sent.start and sp[1] <= sent.end]
        if len(sent_spans) < 2:
            continue
        for tok in sent:
            if tok.pos_ != "VERB":
                continue
            rel = schema.VERB_TO_RELATION.get(tok.lemma_.lower())
            if not rel:
                continue
            before = [sp for sp in sent_spans if sp[1] <= tok.i]
            after = [sp for sp in sent_spans if sp[0] > tok.i]
            if not before or not after:
                continue
            subj, obj = before[-1][2], after[0][2]
            passive = any(c.dep_ in {"auxpass", "agent"} for c in tok.children) or (
                tok.i + 1 < len(doc) and doc[tok.i + 1].lower_ == "by")
            if passive:
                subj, obj = obj, subj
            graph.add_relation(subj, obj, rel, sent.text.strip())
    return graph


# ── Public API with caching ──────────────────────────────────────────────────
_cache_lock = threading.Lock()


def _cache_path():
    return get_settings().storage_dir / "extraction_cache.json"


def _load_cache() -> dict:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    _cache_path().write_text(json.dumps(cache), encoding="utf-8")


def extract(text: str, known: dict[str, str], prefer_llm: Optional[bool] = None) -> ExtractedGraph:
    """
    Extract entities + relations from `text`.
    LLM first (if configured), spaCy as fallback, gazetteer linking always.
    """
    s = get_settings()
    use_llm = s.llm_enabled and s.use_llm_extraction if prefer_llm is None else (prefer_llm and s.llm_enabled)
    method = "llm" if use_llm else "spacy"
    def _key(m: str) -> str:
        return hashlib.sha256(f"{m}|{sorted(known)}|{text}".encode()).hexdigest()

    cache_key = _key(method)

    with _cache_lock:
        cache = _load_cache()
    if cache_key in cache:
        g = ExtractedGraph(**cache[cache_key])
    else:
        g = None
        if use_llm:
            try:
                g = extract_llm(text, known)
            except Exception as exc:
                print(f"[extraction] LLM extraction failed, falling back to spaCy: {exc}")
        if g is None:
            try:
                g = extract_spacy(text, known)
            except Exception as exc:
                print(f"[extraction] spaCy unavailable: {exc}")
                g = ExtractedGraph(method="gazetteer")
        if g.method != "gazetteer":
            with _cache_lock:
                cache = _load_cache()
                cache[_key(g.method)] = {"entities": g.entities, "relations": g.relations, "method": g.method}
                _save_cache(cache)

    gazetteer_link(text, g, known)
    return g
