"""AI Assistant: a chat page grounded in the app's own backend outputs (see ``src/chatbot``)."""

from __future__ import annotations

import streamlit as st

from src.chatbot import llm, prompts, service
from src.ui import components as ui, theme

MESSAGES = "assistant_messages"
PENDING = "assistant_pending"
SOURCE_LABELS = {service.SOURCE_AI: "AI answer", service.SOURCE_FAQ: "Built-in FAQ (not generative AI)",
                 service.SOURCE_GUARD: "Safety notice"}

_CSS = f"""
<style>
.st-key-assistant-page [data-testid="stChatMessage"] {{
  background:{theme.SURFACE}; border:1px solid {theme.BORDER_SOFT}; border-radius:16px; padding:.9rem 1.1rem; margin-bottom:.6rem;
}}
.st-key-assistant-page [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  background:{theme.TEAL_SOFT}; border-color:{theme.SAGE};
}}
.st-key-assistant-page [class*="st-key-sq-"] button {{
  background:{theme.SURFACE}; border:1px solid {theme.BORDER}; border-radius:14px; color:{theme.INK};
  text-align:left; justify-content:flex-start; min-height:3.1rem; font-weight:500; font-size:.86rem; line-height:1.3;
}}
.st-key-assistant-page [class*="st-key-sq-"] button:hover {{ border-color:{theme.PRIMARY}; background:{theme.TEAL_SOFT}; color:{theme.PRIMARY}; }}
.mt-ai-source {{ margin-top:.35rem; font-size:.72rem; color:{theme.MUTED}; }}
.mt-ai-mode {{ display:inline-flex; gap:.45rem; align-items:center; padding:.3rem .75rem; border-radius:999px;
  background:{theme.TEAL_SOFT}; color:{theme.PRIMARY}; font-size:.78rem; font-weight:500; }}
.mt-ai-mode.faq {{ background:{theme.AMBER_SOFT}; color:{theme.AMBER}; }}
</style>
"""

SETUP_HELP = """Add this to `.streamlit/secrets.toml` locally, or to **App settings → Secrets** on Streamlit Community Cloud:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-2.5-flash"   # optional
```
OpenAI remains an optional alternative (`[openai] api_key = "sk-..."`), used only if no Gemini key is set."""


def _ask(question: str) -> None:
    st.session_state[PENDING] = question


def _clear() -> None:
    st.session_state[MESSAGES] = []
    st.session_state.pop(PENDING, None)


def _show(role: str, content: str, source: str = "", notice: str = "") -> None:
    with st.chat_message(role):
        if notice:
            st.warning(notice, icon=":material/info:")
        st.markdown(content)
        if source in SOURCE_LABELS:
            st.markdown(f'<div class="mt-ai-source">{SOURCE_LABELS[source]}</div>', unsafe_allow_html=True)


def _suggestions() -> None:
    cols = st.columns(3)
    for i, q in enumerate(prompts.SUGGESTED_QUESTIONS):
        with cols[i % 3]:
            st.button(q, key=f"sq-{i}", on_click=_ask, args=(q,), width="stretch")


def render() -> None:
    ui.md(ui.page_header(
        "AI Assistant",
        "Ask about beds, forecasts, alerts, the Digital Twin, What-If and the ML models.",
        eyebrow="Ask",
    ))
    st.markdown(_CSS, unsafe_allow_html=True)
    config = llm.config_from_secrets()
    messages: list[dict] = st.session_state.setdefault(MESSAGES, [])

    with st.container(key="assistant-page"):
        if config:
            st.markdown(f'<span class="mt-ai-mode">AI model: {config.provider_label} · {config.model} · answers use live MediTwin values</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="mt-ai-mode faq">AI model not configured · limited FAQ mode</span>',
                        unsafe_allow_html=True)
            with st.expander("How to enable the AI assistant"):
                st.markdown(SETUP_HELP)

        _show("assistant", prompts.WELCOME)
        if messages:
            with st.expander("Suggested questions"):
                _suggestions()
            st.button("Clear chat", icon=":material/delete:", key="assistant-clear", on_click=_clear)
        else:
            st.markdown("**Try asking**")
            _suggestions()

        for m in messages:
            _show(m["role"], m["content"], m.get("source", ""), m.get("notice", ""))

    typed = st.chat_input("Ask about the hospital or the models…", key="assistant-input")
    question = st.session_state.pop(PENDING, None) or typed
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    with st.container(key="assistant-live"):
        _show("user", question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the latest MediTwin values…"):
                reply = service.answer(question, history, config=config)
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": reply.text, "source": reply.source,
                     "notice": reply.notice})
    st.rerun()
