"""LLM clients over plain HTTPS (``requests``; no SDK): Google Gemini (primary) and OpenAI (optional).

Keys come from the environment or Streamlit secrets only and are never logged, put in a URL, or
included in error messages. Every failure becomes an :class:`LLMError` with a message safe to show.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import requests

GEMINI, OPENAI = "gemini", "openai"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODELS = {GEMINI: "gemini-2.5-flash", OPENAI: "gpt-4o-mini"}
TIMEOUT_SECONDS = 45
MAX_OUTPUT_TOKENS = 1500  # Gemini "thinking" tokens count toward this cap, so keep headroom


class LLMError(Exception):
    """A failure with a message that is safe to show to the user."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    model: str = DEFAULT_MODELS[GEMINI]
    provider: str = GEMINI

    @property
    def provider_label(self) -> str:
        return "Google Gemini" if self.provider == GEMINI else "OpenAI"

    def __repr__(self) -> str:  # never leak the key through logs or tracebacks
        return f"LLMConfig(provider={self.provider!r}, model={self.model!r}, api_key=<hidden>)"


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def _get(mapping: Any, key: str) -> Any:
    try:
        return mapping.get(key) if hasattr(mapping, "get") else None
    except Exception:
        return None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def config_from_secrets(secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> LLMConfig | None:
    """Gemini first (``GEMINI_API_KEY`` in secrets or the environment), then OpenAI. ``None`` = not configured."""
    env = os.environ if env is None else env
    try:
        if secrets is None:
            import streamlit as st

            secrets = st.secrets
        _ = _get(secrets, "GEMINI_API_KEY")  # touching it surfaces a missing secrets file here
    except Exception:  # no secrets file or unreadable: fall back to the environment only
        secrets = {}

    gem_sec, oai_sec = _get(secrets, "gemini"), _get(secrets, "openai")
    key = _clean(_get(secrets, "GEMINI_API_KEY") or _get(gem_sec, "api_key") or env.get("GEMINI_API_KEY"))
    if key:
        model = _clean(_get(secrets, "GEMINI_MODEL") or _get(gem_sec, "model") or env.get("GEMINI_MODEL"))
        return LLMConfig(key, model or DEFAULT_MODELS[GEMINI], GEMINI)

    key = _clean(_get(oai_sec, "api_key") or _get(secrets, "OPENAI_API_KEY") or env.get("OPENAI_API_KEY"))
    if key:
        model = _clean(_get(oai_sec, "model") or _get(secrets, "OPENAI_MODEL") or env.get("OPENAI_MODEL"))
        return LLMConfig(key, model or DEFAULT_MODELS[OPENAI], OPENAI)
    return None


# ----------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------
def _send(post, url: str, headers: dict, body: dict) -> Any:
    try:
        return post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
    except requests.Timeout as exc:
        raise LLMError("The AI service took too long to respond.") from None
    except requests.RequestException:
        raise LLMError("Could not reach the AI service.") from None


def _error_message(resp: Any) -> str:
    try:
        err = resp.json().get("error", {})
        return str(err.get("message", "")) if isinstance(err, dict) else ""
    except Exception:
        return ""


def _raise_for_status(resp: Any) -> None:
    status = resp.status_code
    if status < 400:
        return
    detail = _error_message(resp).lower()
    if status in (401, 403) or (status == 400 and "api key" in detail):
        raise LLMError("The AI service rejected the configured API key.")
    if status == 404:
        raise LLMError("The configured AI model was not found. Check the model name in the secrets.")
    if status == 429:
        raise LLMError("The AI service is rate-limited or the API quota is used up.")
    if status == 400:
        raise LLMError("The AI service could not process the request (HTTP 400).")
    raise LLMError(f"The AI service returned an error (HTTP {status}).")


# ----------------------------------------------------------------------
# Gemini (generateContent)
# ----------------------------------------------------------------------
def gemini_request(config: LLMConfig, messages: list[dict[str, str]]) -> tuple[str, dict, dict]:
    """(url, headers, body) for ``models/{model}:generateContent``. The key goes in a header, not the URL."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents: list[dict] = []
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            continue
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            continue  # the conversation must start with a user turn
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    body: dict[str, Any] = {"contents": contents, "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = GEMINI_URL.format(model=config.model)
    return url, {"x-goog-api-key": config.api_key, "Content-Type": "application/json"}, body


def parse_gemini(data: Any) -> str:
    """Text of the first candidate, or an :class:`LLMError` describing why there is none."""
    if not isinstance(data, dict):
        raise LLMError("The AI service returned an unexpected response.")
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        if reason:
            raise LLMError("The AI service declined to answer this question.")
        raise LLMError("The AI service returned an unexpected response.")
    cand = candidates[0] if isinstance(candidates[0], dict) else {}
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
    finish = cand.get("finishReason")
    if not text:
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"):
            raise LLMError("The AI service declined to answer this question.")
        if finish == "MAX_TOKENS":
            raise LLMError("The AI service ran out of output space before answering.")
        raise LLMError("The AI service returned an empty answer.")
    if finish == "MAX_TOKENS":
        text += "\n\n_(The answer was cut short.)_"
    return text


def _chat_gemini(config: LLMConfig, messages: list[dict[str, str]], post) -> str:
    url, headers, body = gemini_request(config, messages)
    resp = _send(post, url, headers, body)
    _raise_for_status(resp)
    try:
        data = resp.json()
    except ValueError:
        raise LLMError("The AI service returned an unexpected response.") from None
    return parse_gemini(data)


# ----------------------------------------------------------------------
# OpenAI (optional)
# ----------------------------------------------------------------------
def _chat_openai(config: LLMConfig, messages: list[dict[str, str]], post) -> str:
    resp = _send(
        post, OPENAI_URL,
        {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        {"model": config.model, "messages": messages, "max_completion_tokens": MAX_OUTPUT_TOKENS},
    )
    _raise_for_status(resp)
    try:
        text = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("The AI service returned an unexpected response.") from None
    text = (text or "").strip()
    if not text:
        raise LLMError("The AI service returned an empty answer.")
    return text


def chat(config: LLMConfig, messages: list[dict[str, str]], post=requests.post) -> str:
    """Send the conversation (OpenAI-style role/content messages) and return the reply text."""
    if config.provider == OPENAI:
        return _chat_openai(config, messages, post)
    return _chat_gemini(config, messages, post)
