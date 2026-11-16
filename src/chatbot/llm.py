"""Minimal OpenAI Chat Completions client (plain HTTPS via ``requests``; no extra dependency).

The key is read from Streamlit secrets only and is never logged or included in error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import requests

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"  # small, low-cost; override with openai.model in secrets
TIMEOUT_SECONDS = 45
MAX_OUTPUT_TOKENS = 600


class LLMError(Exception):
    """A failure with a message that is safe to show to the user."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    model: str = DEFAULT_MODEL

    def __repr__(self) -> str:  # never leak the key through logs or tracebacks
        return f"LLMConfig(model={self.model!r}, api_key=<hidden>)"


def config_from_secrets(secrets: Mapping[str, Any] | None = None) -> LLMConfig | None:
    """Read ``[openai] api_key`` (or a top-level ``OPENAI_API_KEY``). ``None`` when not configured."""
    try:
        if secrets is None:
            import streamlit as st

            secrets = st.secrets
        section = secrets.get("openai") if hasattr(secrets, "get") else None
        key = (section.get("api_key") if section is not None and hasattr(section, "get") else None) \
            or secrets.get("OPENAI_API_KEY")
        model = (section.get("model") if section is not None and hasattr(section, "get") else None) \
            or secrets.get("OPENAI_MODEL") or DEFAULT_MODEL
    except Exception:  # no secrets file, or an unreadable one: treat as not configured
        return None
    key = str(key or "").strip()
    return LLMConfig(api_key=key, model=str(model).strip()) if key else None


def chat(config: LLMConfig, messages: list[dict[str, str]], post=requests.post) -> str:
    """Send the conversation and return the reply text, or raise :class:`LLMError`."""
    try:
        resp = post(
            API_URL,
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json={"model": config.model, "messages": messages, "max_completion_tokens": MAX_OUTPUT_TOKENS},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise LLMError("The AI service took too long to respond.") from exc
    except requests.RequestException as exc:
        raise LLMError("Could not reach the AI service.") from exc

    status = resp.status_code
    if status in (401, 403):
        raise LLMError("The AI service rejected the configured API key.")
    if status == 429:
        raise LLMError("The AI service is rate-limited or the API quota is used up.")
    if status == 404:
        raise LLMError("The configured AI model was not found. Check `openai.model` in the secrets.")
    if status >= 400:
        raise LLMError(f"The AI service returned an error (HTTP {status}).")
    try:
        text = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError("The AI service returned an unexpected response.") from exc
    text = (text or "").strip()
    if not text:
        raise LLMError("The AI service returned an empty answer.")
    return text
