"""
llm.py — thin wrapper around Google Gemini (google-genai SDK).

* Uses the current `google-genai` SDK (the old `google-generativeai` package and
  the gemini-1.5 models are retired).
* Model is configurable with GEMINI_MODEL; if it is unavailable we fall back
  through a short list of current Flash models.
* Retries on rate limits (HTTP 429) with exponential back-off — the free tier
  has low requests-per-minute limits.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional, Type

from .config import get_settings

FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    s = get_settings()
    if not s.gemini_api_key:
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    from google import genai
    from google.genai import types

    return genai.Client(api_key=s.gemini_api_key, http_options=types.HttpOptions(timeout=90_000))


_working_model: Optional[str] = None


def _candidate_models() -> list[str]:
    preferred = get_settings().gemini_model
    seen, out = set(), []
    for m in ([_working_model] if _working_model else []) + [preferred] + FALLBACK_MODELS:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def generate(prompt: str, *, schema: Optional[Type] = None, temperature: float = 0.2,
             system: Optional[str] = None, max_retries: int = 4):
    """
    Call Gemini. If `schema` (a pydantic model) is given, the model is forced to
    return JSON matching it and the parsed object is returned; otherwise text.
    """
    global _working_model
    from google.genai import errors, types

    client = _client()
    cfg = dict(temperature=temperature)
    if system:
        cfg["system_instruction"] = system
    if schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = schema
    config = types.GenerateContentConfig(**cfg)

    last_error: Exception | None = None
    for model in _candidate_models():
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                _working_model = model
                if schema is not None:
                    if resp.parsed is not None:
                        return resp.parsed
                    return schema.model_validate_json(resp.text)
                return (resp.text or "").strip()
            except errors.APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code == 429 or (code and code >= 500):
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                if code in (400, 404) and "model" in str(exc).lower():
                    break  # try the next model name
                raise LLMUnavailable(str(exc)) from exc
            except Exception as exc:  # network errors, JSON errors …
                last_error = exc
                time.sleep(1 + attempt)
        # next model
    raise LLMUnavailable(f"Gemini call failed: {last_error}")


def active_model() -> str:
    return _working_model or get_settings().gemini_model
