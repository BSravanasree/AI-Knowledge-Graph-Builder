"""
schema.py — Knowledge Graph schema (Milestone 1: "define knowledge graph schema").

The graph mixes two kinds of knowledge:

  STRUCTURED (from the transactions table and the fraud-case database)
    (:Transaction)-[:AT_MERCHANT]->(:MerchantCategory)
    (:Transaction)-[:IN_TIMESLOT]->(:TimeSlot)
    (:Transaction)-[:BY_SEGMENT]->(:CustomerSegment)
    (:Transaction)-[:HAS_RISK_FACTOR]->(:RiskFactor)
    (:Transaction)-[:FLAGGED_AS]->(:FraudStatus)
    (:Person)-[:INVESTIGATES]->(:Transaction)        <- analyst on a fraud case

  UNSTRUCTURED (LLM / NLP extraction from documents and emails)
    (:Document)-[:MENTIONS]->(any entity)
    (:Organization|:Person|:Location|...)-[:<RELATION>]->(...)

Because document entities are resolved onto the same keys as structured
nodes (e.g. a memo that mentions "Electronics" links to the same
MerchantCategory node the transactions point at), the graph exposes
cross-source relationships that no single source shows on its own.
"""

from __future__ import annotations

import re

# ── Node labels ────────────────────────────────────────────────────────────────
STRUCTURED_LABELS = [
    "Transaction",
    "MerchantCategory",
    "TimeSlot",
    "CustomerSegment",
    "RiskFactor",
    "FraudStatus",
]

# Labels the extractor (LLM or spaCy) is allowed to create.
ENTITY_LABELS = [
    "Person",
    "Organization",
    "Location",
    "Product",
    "Event",
    "Policy",
    "MerchantCategory",
    "RiskFactor",
    "Concept",
]

DOCUMENT_LABEL = "Document"

ALL_LABELS = sorted(set(STRUCTURED_LABELS + ENTITY_LABELS + [DOCUMENT_LABEL]))

# Colours used by the dashboard (one per label).
LABEL_COLORS = {
    "Transaction": "#8da2b8",
    "MerchantCategory": "#f2a541",
    "TimeSlot": "#7fb7be",
    "CustomerSegment": "#b39ddb",
    "RiskFactor": "#e4572e",
    "FraudStatus": "#c1121f",
    "Document": "#4f6d7a",
    "Person": "#2a9d8f",
    "Organization": "#3a86ff",
    "Location": "#90be6d",
    "Product": "#f28482",
    "Event": "#ffb4a2",
    "Policy": "#6d597a",
    "Concept": "#adb5bd",
}

# ── Relationship types ────────────────────────────────────────────────────────
STRUCTURED_RELATIONS = [
    "AT_MERCHANT",
    "IN_TIMESLOT",
    "BY_SEGMENT",
    "HAS_RISK_FACTOR",
    "FLAGGED_AS",
    "INVESTIGATES",
    "LOCATED_IN",
    "MENTIONS",
    "AUTHORED",
]

# Canonical relation vocabulary for extracted facts. The LLM is asked to use
# these; anything else is normalised by `normalise_relation` below.
EXTRACTED_RELATIONS = [
    "PARTNERED_WITH",
    "WORKS_FOR",
    "REPORTS_TO",
    "LOCATED_IN",
    "OWNS",
    "SUPPLIES",
    "ACQUIRED",
    "INVESTIGATES",
    "DETECTED",
    "AFFECTS",
    "TARGETS",
    "MITIGATES",
    "RESPONSIBLE_FOR",
    "APPROVED",
    "USES",
    "INCREASES_RISK_OF",
    "RELATED_TO",
]

# Verb lemma → canonical relation (used by the spaCy fallback extractor).
VERB_TO_RELATION = {
    "partner": "PARTNERED_WITH",
    "collaborate": "PARTNERED_WITH",
    "work": "WORKS_FOR",
    "join": "WORKS_FOR",
    "locate": "LOCATED_IN",
    "base": "LOCATED_IN",
    "own": "OWNS",
    "supply": "SUPPLIES",
    "provide": "SUPPLIES",
    "acquire": "ACQUIRED",
    "buy": "ACQUIRED",
    "investigate": "INVESTIGATES",
    "review": "INVESTIGATES",
    "detect": "DETECTED",
    "flag": "DETECTED",
    "identify": "DETECTED",
    "affect": "AFFECTS",
    "hit": "AFFECTS",
    "target": "TARGETS",
    "mitigate": "MITIGATES",
    "reduce": "MITIGATES",
    "block": "MITIGATES",
    "lead": "RESPONSIBLE_FOR",
    "manage": "RESPONSIBLE_FOR",
    "approve": "APPROVED",
    "sign": "APPROVED",
    "use": "USES",
    "deploy": "USES",
    "adopt": "USES",
}

_REL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,48}$")


def normalise_relation(raw: str) -> str:
    """
    Turn any relation string into a safe, canonical Neo4j relationship type.

    "partnered with" -> "PARTNERED_WITH"; unknown/unsafe -> "RELATED_TO".
    The result only ever contains A-Z, 0-9 and "_", so it is safe to place in
    a Cypher query (relationship types cannot be passed as parameters).
    """
    rel = re.sub(r"[^A-Za-z0-9]+", "_", (raw or "").strip()).strip("_").upper()
    if rel in EXTRACTED_RELATIONS or rel in STRUCTURED_RELATIONS:
        return rel
    lemma = rel.lower().split("_")[0]
    for verb, canonical in VERB_TO_RELATION.items():
        if lemma.startswith(verb):
            return canonical
    return "RELATED_TO"


def normalise_label(raw: str) -> str:
    """Map free-text entity types (ORG, GPE, company…) onto schema labels."""
    t = (raw or "").strip().lower()
    mapping = {
        "person": "Person", "per": "Person", "people": "Person", "analyst": "Person",
        "organization": "Organization", "organisation": "Organization", "org": "Organization",
        "company": "Organization", "vendor": "Organization", "bank": "Organization",
        "team": "Organization", "department": "Organization",
        "location": "Location", "loc": "Location", "gpe": "Location", "city": "Location",
        "country": "Location", "place": "Location", "fac": "Location",
        "product": "Product", "system": "Product", "tool": "Product", "software": "Product",
        "event": "Event", "incident": "Event", "attack": "Event",
        "policy": "Policy", "rule": "Policy", "regulation": "Policy", "law": "Policy",
        "merchantcategory": "MerchantCategory", "merchant_category": "MerchantCategory",
        "merchant category": "MerchantCategory", "merchant": "MerchantCategory",
        "riskfactor": "RiskFactor", "risk_factor": "RiskFactor", "risk factor": "RiskFactor",
        "risk": "RiskFactor",
    }
    if t in mapping:
        return mapping[t]
    for label in ENTITY_LABELS:
        if t == label.lower():
            return label
    return "Concept"


def make_key(label: str, name: str) -> str:
    """Stable unique id for a node: 'organization:acme corp'."""
    clean = re.sub(r"\s+", " ", str(name)).strip().lower()
    return f"{label.lower()}:{clean}"


# ── Business rules used to derive structured nodes ────────────────────────────
def time_slot(hour: int) -> str:
    if hour <= 5:
        return "Night (00-05)"
    if hour <= 11:
        return "Morning (06-11)"
    if hour <= 17:
        return "Afternoon (12-17)"
    return "Evening (18-23)"


def customer_segment(age: int) -> str:
    if age <= 25:
        return "Age 18-25"
    if age <= 35:
        return "Age 26-35"
    if age <= 50:
        return "Age 36-50"
    if age <= 65:
        return "Age 51-65"
    return "Age 66+"


LOW_TRUST_THRESHOLD = 40   # device_trust_score below this is "low"
HIGH_VELOCITY_THRESHOLD = 5  # 5+ transactions in 24h is "high velocity"

RISK_FACTORS = {
    "Foreign Transaction": "Card used outside the cardholder's home country.",
    "Location Mismatch": "Transaction location differs from the device/billing location.",
    "Low Device Trust": f"Device trust score below {LOW_TRUST_THRESHOLD}.",
    "High Velocity": f"{HIGH_VELOCITY_THRESHOLD}+ transactions by the card in the last 24 hours.",
}


def risk_factors_for(row) -> list[str]:
    out = []
    if int(row["foreign_transaction"]) == 1:
        out.append("Foreign Transaction")
    if int(row["location_mismatch"]) == 1:
        out.append("Location Mismatch")
    if int(row["device_trust_score"]) < LOW_TRUST_THRESHOLD:
        out.append("Low Device Trust")
    if int(row["velocity_last_24h"]) >= HIGH_VELOCITY_THRESHOLD:
        out.append("High Velocity")
    return out
