"""
config.py — one place for every setting.

Values are read in this order (first hit wins):
  1. Real environment variables
  2. A local `.env` file (for running on your laptop)
  3. Streamlit secrets (`.streamlit/secrets.toml` locally, or the
     "Secrets" box on Streamlit Community Cloud when deployed)
  4. The defaults below
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _from_streamlit_secrets(name: str):
    """Read a key from st.secrets without crashing when no secrets file exists."""
    try:
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value in (None, ""):
        value = _from_streamlit_secrets(name)
    if value in (None, ""):
        return default
    return str(value).strip()


@dataclass
class Settings:
    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    sources_dir: Path = PROJECT_ROOT / "data" / "sources"
    storage_dir: Path = field(  # generated artefacts (gitignored)
        default_factory=lambda: Path(get_setting("STORAGE_DIR", str(PROJECT_ROOT / "storage"))))

    # ── Graph database ───────────────────────────────────────────────────────
    graph_backend: str = field(default_factory=lambda: get_setting("GRAPH_BACKEND", "auto"))
    neo4j_uri: str = field(default_factory=lambda: get_setting("NEO4J_URI"))
    neo4j_user: str = field(default_factory=lambda: get_setting("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: get_setting("NEO4J_PASSWORD"))
    neo4j_database: str = field(default_factory=lambda: get_setting("NEO4J_DATABASE"))

    # ── LLM (Google Gemini) ──────────────────────────────────────────────────
    gemini_api_key: str = field(default_factory=lambda: get_setting("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: get_setting("GEMINI_MODEL", "gemini-3.6-flash"))
    use_llm_extraction: bool = field(
        default_factory=lambda: get_setting("USE_LLM_EXTRACTION", "true").lower() == "true"
    )

    # ── Embeddings / RAG ─────────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: get_setting("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_backend: str = field(default_factory=lambda: get_setting("EMBEDDING_BACKEND", "auto"))
    top_k: int = field(default_factory=lambda: int(get_setting("RAG_TOP_K", "6")))

    # ── App ──────────────────────────────────────────────────────────────────
    app_password: str = field(default_factory=lambda: get_setting("APP_PASSWORD"))

    @property
    def transactions_csv(self) -> Path:
        return self.data_dir / "creditcard.csv"

    @property
    def cases_db(self) -> Path:
        return self.data_dir / "enterprise.db"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def neo4j_configured(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)


def get_settings() -> Settings:
    s = Settings()
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    return s
