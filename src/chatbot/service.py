"""Orchestration: guardrails -> live context -> LLM (or FAQ fallback). Never raises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import requests

from src.chatbot import context as ctx_mod, faq, guardrails, llm, prompts

HISTORY_TURNS = 6  # messages of earlier conversation sent to the model

SOURCE_AI, SOURCE_FAQ, SOURCE_GUARD = "ai", "faq", "guardrail"

NO_KEY_NOTICE = (
    "The AI model is not configured, so I'm using the limited built-in FAQ instead. "
    "An admin can enable the full assistant by adding an OpenAI API key to the Streamlit secrets."
)


@dataclass(frozen=True)
class Reply:
    text: str
    source: str  # ai | faq | guardrail
    notice: str = ""  # shown above the text (configuration problem or service error)


def _history_messages(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    keep = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    return [{"role": m["role"], "content": str(m["content"])[:2000]} for m in keep[-HISTORY_TURNS:]]


def answer(
    question: str,
    history: list[dict[str, str]] | None = None,
    config: llm.LLMConfig | None = None,
    backend: Any | None = None,
    post: Callable[..., requests.Response] = requests.post,
) -> Reply:
    q = guardrails.clean_question(question)
    if not q:
        return Reply(guardrails.EMPTY_REPLY, SOURCE_GUARD)
    if guardrails.is_clinical_request(q):
        return Reply(guardrails.CLINICAL_REFUSAL, SOURCE_GUARD)

    try:
        context = ctx_mod.build_context(q, backend=backend)
    except Exception as exc:  # a backend problem must not crash the page
        context = {"project": "", "unavailable": [f"Live data could not be loaded ({type(exc).__name__})."]}

    if config is None:
        text, _ = faq.faq_answer(q, context)
        return Reply(text, SOURCE_FAQ, notice=NO_KEY_NOTICE)

    messages = [{"role": "system", "content": prompts.build_system_prompt(context)},
                *_history_messages(history), {"role": "user", "content": q}]
    try:
        return Reply(llm.chat(config, messages, post=post), SOURCE_AI)
    except llm.LLMError as exc:
        text, _ = faq.faq_answer(q, context)
        return Reply(text, SOURCE_FAQ, notice=f"{exc} Showing the limited built-in FAQ instead.")
    except Exception:  # unexpected: still degrade gracefully
        text, _ = faq.faq_answer(q, context)
        return Reply(text, SOURCE_FAQ, notice="Something went wrong with the AI service. Showing the limited built-in FAQ instead.")
