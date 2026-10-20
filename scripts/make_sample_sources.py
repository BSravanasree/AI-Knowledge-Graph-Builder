"""
make_sample_sources.py — generates the SYNTHETIC enterprise sources used in the demo.

Run once (already done; outputs are committed):
    pip install fpdf2
    python scripts/make_sample_sources.py

Creates
  data/sources/documents/*.pdf|.docx|.md|.txt   (reports, policies, incident write-ups)
  data/sources/emails/*.eml                     (internal email threads)
  data/enterprise.db                            (SQLite "case management" database)

All companies and people are fictional. Numbers quoted in the documents match
the patterns actually present in data/creditcard.csv, so RAG answers can be
cross-checked against the transaction graph.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from email.message import EmailMessage
from email.utils import format_datetime
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "sources" / "documents"
MAILS = ROOT / "data" / "sources" / "emails"
DOCS.mkdir(parents=True, exist_ok=True)
MAILS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Documents
# ─────────────────────────────────────────────────────────────────────────────
Q3_REVIEW = [
    ("Q3 Fraud Risk Review - Meridian Card Services", None),
    ("Prepared by Priya Raman, Head of Fraud Risk, Chennai office. Reviewed by Daniel Brooks, CISO.", None),
    ("1. Summary", "h"),
    ("Meridian Card Services processed 10,000 card transactions in the review window. "
     "151 transactions (1.51%) were confirmed as fraud. Fraud is heavily concentrated in a few "
     "conditions rather than spread evenly across the portfolio.", None),
    ("2. Key risk drivers", "h"),
    ("Night-time window: transactions between 00:00 and 05:59 have a fraud rate of about 5.1%, "
     "compared with roughly 0.3% to 0.4% in the morning, afternoon and evening. 124 of the 151 "
     "fraud cases happened at night.", None),
    ("Low device trust: devices with a Sentinel DeviceID trust score below 40 show a 6.2% fraud "
     "rate, versus 0.3% for devices scoring above 70.", None),
    ("Foreign transactions show an 8.4% fraud rate and location mismatches also show an 8.4% "
     "fraud rate, roughly ten times the domestic baseline of 0.8%.", None),
    ("High velocity: cards with 5 or more transactions in 24 hours have a fraud rate near 10%.", None),
    ("3. Merchant categories", "h"),
    ("Grocery has the highest fraud rate at 2.0% (39 cases), followed by Food at 1.7% (35 cases) "
     "and Travel at 1.5% (29 cases). Electronics and Clothing are the lowest at about 1.2% each. "
     "This contradicts the common assumption that Electronics is our riskiest category.", None),
    ("4. Customer segments", "h"),
    ("Fraud rates are flat across age groups (1.3% to 1.6%), so age is not a useful risk signal "
     "on its own.", None),
    ("5. Recommendations", "h"),
    ("a) Apply step-up authentication to night-time transactions that also carry a Low Device "
     "Trust or Foreign Transaction flag.", None),
    ("b) Extend the partnership with Sentinel DeviceID Ltd to cover all mobile channels.", None),
    ("c) Keep velocity rule VR-5 in force; it targets the High Velocity risk factor.", None),
    ("d) Arjun Mehta will lead a deep-dive into Grocery fraud and location mismatch patterns.", None),
]


def make_pdf(path: Path, blocks):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for i, (text, kind) in enumerate(blocks):
        if i == 0:
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif kind == "h":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    pdf.output(str(path))


VENDOR_AGREEMENT = [
    ("Vendor Agreement Summary - Sentinel DeviceID Ltd", "title"),
    ("Contract reference: VND-2026-014. Status: Active.", None),
    ("Parties", "h"),
    ("Meridian Card Services partnered with Sentinel DeviceID Ltd, a device-fingerprinting "
     "company headquartered in Singapore. Sentinel DeviceID supplies the device trust score "
     "that Meridian uses in real-time fraud scoring.", None),
    ("Scope", "h"),
    ("Sentinel DeviceID provides a trust score from 0 to 99 for every card-not-present "
     "transaction. Scores below 40 are treated as Low Device Trust by the fraud engine. "
     "The integration is owned by Kavya Iyer, Lead Data Engineer.", None),
    ("Approvals", "h"),
    ("The agreement was approved by Daniel Brooks, Chief Information Security Officer, and "
     "signed off by Meera Nair, Head of Compliance, after a data-protection review.", None),
    ("Rollout", "h"),
    ("Phase 1 covered the Chennai and Singapore processing centres. Phase 2 extends coverage "
     "to the London office and all mobile channels, as recommended in the Q3 Fraud Risk Review.", None),
]


def make_docx(path: Path, blocks):
    from docx import Document

    doc = Document()
    for text, kind in blocks:
        if kind == "title":
            doc.add_heading(text, level=0)
        elif kind == "h":
            doc.add_heading(text, level=1)
        else:
            doc.add_paragraph(text)
    doc.save(str(path))


INCIDENT_MD = """# Incident Report INC-2041: Card-Testing Attack on Travel Merchants

**Severity:** High  **Status:** Resolved  **Owner:** Arjun Mehta, Fraud Analyst

## What happened
Between 01:00 and 04:30 on three consecutive nights, an automated card-testing attack
targeted Travel and Electronics merchants. Most attempts originated from IP ranges in
Dubai and were marked as Foreign Transaction with Location Mismatch.

## Detection
Arjun Mehta detected the attack when the fraud dashboard showed a cluster of
night-time transactions linked to the High Velocity and Low Device Trust risk factors.
Sentinel DeviceID scores for the attacking devices were below 20.

## Response
Velocity rule VR-5 blocked further attempts once a card reached five transactions in
24 hours. Harbor Trust Bank, our issuing partner, reissued 212 affected cards.

## Lessons learned
Night-time transactions combining Foreign Transaction and Low Device Trust should
trigger step-up authentication automatically. Priya Raman approved this change.
"""

POLICY_TXT = """Fraud Control Policy VR-5: Transaction Velocity Limit

Policy owner: Meera Nair, Head of Compliance
Applies to: all Meridian Card Services card products

1. Purpose
Velocity rule VR-5 mitigates High Velocity fraud, where a compromised card is used many
times in a short period.

2. Rule
If a card performs 5 or more transactions within 24 hours, any further transaction is
held for review by the fraud operations team in Chennai.

3. Evidence
Cards with 5 or more transactions in 24 hours show a fraud rate close to 10 percent,
against a portfolio average of 1.5 percent (Q3 Fraud Risk Review).

4. Exceptions
Travel merchants may request a temporary exception during peak season. Exceptions must
be approved by Priya Raman.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Emails
# ─────────────────────────────────────────────────────────────────────────────
EMAILS = [
    dict(
        file="2026-07-02_night_fraud_spike.eml",
        sender="Priya Raman <priya.raman@meridiancards.example>",
        to="Fraud Team <fraud-team@meridiancards.example>",
        subject="Night-time fraud spike - action needed",
        when=datetime(2026, 7, 2, 9, 15, tzinfo=timezone.utc),
        body=(
            "Team,\n\nThe overnight report shows most of this month's confirmed fraud happened "
            "between midnight and 6 am. Grocery merchants are over-represented, which is unusual.\n\n"
            "Arjun, please investigate the Grocery cases and check whether they share a Location "
            "Mismatch pattern. Kavya, can you make sure the knowledge graph refresh includes the "
            "new Sentinel DeviceID scores?\n\nThanks,\nPriya Raman\nHead of Fraud Risk"
        ),
    ),
    dict(
        file="2026-07-09_grocery_findings.eml",
        sender="Arjun Mehta <arjun.mehta@meridiancards.example>",
        to="Priya Raman <priya.raman@meridiancards.example>",
        subject="RE: Night-time fraud spike - Grocery findings",
        when=datetime(2026, 7, 9, 16, 40, tzinfo=timezone.utc),
        body=(
            "Hi Priya,\n\nFindings from the Grocery deep-dive:\n"
            "- Most fraudulent Grocery transactions carry the Location Mismatch flag.\n"
            "- Many also come from devices with low Sentinel DeviceID trust scores.\n"
            "- Harbor Trust Bank reported the same pattern across their Chennai portfolio last month.\n\n"
            "I suggest we share indicators with Harbor Trust Bank and add Grocery to the step-up "
            "authentication pilot.\n\nRegards,\nArjun Mehta\nFraud Analyst\n\n"
            "> On 2 Jul, Priya Raman wrote:\n> Arjun, please investigate the Grocery cases"
        ),
    ),
    dict(
        file="2026-07-15_pipeline_update.eml",
        sender="Kavya Iyer <kavya.iyer@meridiancards.example>",
        to="Priya Raman <priya.raman@meridiancards.example>",
        subject="Knowledge graph pipeline update",
        when=datetime(2026, 7, 15, 11, 5, tzinfo=timezone.utc),
        body=(
            "Hi Priya,\n\nThe nightly pipeline now loads transactions, fraud cases and documents into "
            "the Neo4j knowledge graph. Sentinel DeviceID trust scores are part of every transaction "
            "node. We also signed a data-sharing agreement with Harbor Trust Bank so their fraud "
            "indicators can be added next quarter.\n\nKavya Iyer\nLead Data Engineer"
        ),
    ),
    dict(
        file="2026-08-01_sentinel_rollout_london.eml",
        sender="Daniel Brooks <daniel.brooks@meridiancards.example>",
        to="All Staff <all@meridiancards.example>",
        subject="Sentinel DeviceID rollout to London",
        when=datetime(2026, 8, 1, 8, 0, tzinfo=timezone.utc),
        body=(
            "Hello everyone,\n\nFollowing the Q3 Fraud Risk Review, we are extending Sentinel DeviceID "
            "device fingerprinting to the London office and to all mobile channels from next week. "
            "Meera Nair has completed the compliance review. Please direct questions to Kavya Iyer.\n\n"
            "Daniel Brooks\nChief Information Security Officer"
        ),
    ),
]


def make_emails():
    for e in EMAILS:
        msg = EmailMessage()
        msg["From"] = e["sender"]
        msg["To"] = e["to"]
        msg["Subject"] = e["subject"]
        msg["Date"] = format_datetime(e["when"])
        msg.set_content(e["body"])
        (MAILS / e["file"]).write_bytes(bytes(msg))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Case-management database (SQLite)
# ─────────────────────────────────────────────────────────────────────────────
ANALYSTS = [
    ("Arjun Mehta", "Fraud Operations", "Chennai"),
    ("Rahul Verma", "Fraud Operations", "Chennai"),
    ("Sara Thomas", "Fraud Operations", "Singapore"),
    ("Oliver Grant", "Fraud Operations", "London"),
]


def make_db():
    rng = random.Random(42)
    tx = pd.read_csv(ROOT / "data" / "creditcard.csv")
    fraud = tx[tx["is_fraud"] == 1].reset_index(drop=True)

    db_path = ROOT / "data" / "enterprise.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("CREATE TABLE analysts (name TEXT PRIMARY KEY, team TEXT, location TEXT)")
    cur.executemany("INSERT INTO analysts VALUES (?,?,?)", ANALYSTS)
    cur.execute(
        """CREATE TABLE fraud_cases (
               case_id TEXT PRIMARY KEY,
               transaction_id INTEGER,
               analyst TEXT REFERENCES analysts(name),
               status TEXT,
               opened_on TEXT,
               notes TEXT)"""
    )
    start = date(2026, 7, 1)
    rows = []
    for i, r in fraud.iterrows():
        analyst = rng.choice(ANALYSTS)[0]
        status = rng.choices(["Closed - Confirmed", "Open", "Closed - Recovered"], [6, 2, 2])[0]
        flags = []
        if r.foreign_transaction:
            flags.append("foreign transaction")
        if r.location_mismatch:
            flags.append("location mismatch")
        if r.device_trust_score < 40:
            flags.append(f"low device trust ({r.device_trust_score})")
        if r.velocity_last_24h >= 5:
            flags.append(f"{r.velocity_last_24h} transactions in 24h")
        flag_text = ", ".join(flags) if flags else "no rule flags; reported by cardholder"
        notes = (
            f"{r.merchant_category} purchase of ${r.amount:.2f} at {int(r.transaction_hour):02d}:00. "
            f"Signals: {flag_text}. Assigned to {analyst}."
        )
        rows.append((f"CASE-{1001 + i}", int(r.transaction_id), analyst, status,
                     str(start + timedelta(days=rng.randint(0, 60))), notes))
    cur.executemany("INSERT INTO fraud_cases VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return len(rows)


if __name__ == "__main__":
    make_pdf(DOCS / "Q3_Fraud_Risk_Review.pdf", Q3_REVIEW)
    make_docx(DOCS / "Vendor_Agreement_Sentinel_DeviceID.docx", VENDOR_AGREEMENT)
    (DOCS / "Incident_Report_INC-2041.md").write_text(INCIDENT_MD, encoding="utf-8")
    (DOCS / "Policy_VR-5_Velocity_Limit.txt").write_text(POLICY_TXT, encoding="utf-8")
    make_emails()
    n = make_db()
    print(f"Created 4 documents, {len(EMAILS)} emails and {n} fraud cases.")
