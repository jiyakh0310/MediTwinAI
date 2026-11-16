"""Tests for the AI Assistant: context, FAQ fallback, configuration, API errors, guardrails, page."""

from __future__ import annotations

import json
import unittest
import warnings
from types import SimpleNamespace

import requests

from src.chatbot import context as ctx_mod, faq, guardrails, llm, prompts, service
from src.ui import auth, backend
from src.ui.views import NAV_GROUPS, VIEW_IDS

warnings.filterwarnings("ignore")


def fake_response(status=200, body=None):
    r = SimpleNamespace(status_code=status)
    r.json = lambda: body if body is not None else {}
    return r


def ok_post(text="Hello from the model"):
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return fake_response(200, {"choices": [{"message": {"content": text}}]})

    post.calls = calls
    return post


CONFIG = llm.LLMConfig(api_key="sk-test-secret", model="test-model")


class TestSuggestedQuestions(unittest.TestCase):
    def test_exact_twelve_questions(self):
        self.assertEqual(len(prompts.SUGGESTED_QUESTIONS), 12)
        self.assertEqual(prompts.SUGGESTED_QUESTIONS[0], "What is the current hospital status?")
        self.assertEqual(prompts.SUGGESTED_QUESTIONS[-1], "What happens if patient inflow increases by 20%?")

    def test_every_suggested_question_gets_a_faq_answer(self):
        ctx = ctx_mod.build_context("what if inflow 20%")
        for q in prompts.SUGGESTED_QUESTIONS:
            text, matched = faq.faq_answer(q, ctx)
            self.assertTrue(matched, q)
            self.assertIn("not generative AI", text)


class TestContextRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = ctx_mod.build_context("What happens if patient inflow increases by 20%?")

    def test_live_values_match_the_backend(self):
        state = backend.current_state().value
        today = self.ctx["today"]
        self.assertEqual(self.ctx["data_through"], state.as_of_date)
        self.assertEqual(today["beds"]["occupied"], state.occupied_beds)
        self.assertEqual(today["beds"]["available"], state.available_beds)
        self.assertEqual(today["icu"]["total"], state.total_icu_beds)
        self.assertEqual(today["outpatients_opd"], state.current_opd)
        self.assertEqual(today["medicine_stock_units"], state.medicine_stock)

    def test_forecast_alerts_and_metrics_are_present_and_labelled(self):
        self.assertEqual(len(self.ctx["opd_forecast_7_days"]["days"]), 7)
        self.assertIn("Expected", self.ctx["opd_forecast_7_days"]["type"])
        report, recs = backend.operational_report(1).value
        self.assertEqual(self.ctx["alerts"]["active_count"], len(report.alerts))
        self.assertEqual(len(self.ctx["recommendations"]), len(recs))
        self.assertIn("patient_risk", self.ctx["model_metrics"])
        self.assertEqual(self.ctx["unavailable"], [])

    def test_whatif_results_are_labelled_hypothetical_and_only_added_when_relevant(self):
        results = self.ctx["what_if_presets"]["results"]
        self.assertTrue(results)
        self.assertTrue(all("hypothetical" in r["type"].lower() for r in results))
        self.assertNotIn("what_if_presets", ctx_mod.build_context("How many beds are available?"))

    def test_unavailable_backend_is_reported_not_invented(self):
        broken = SimpleNamespace(
            current_state=lambda: backend.Outcome(error="boom"),
            predicted_state=lambda d: backend.Outcome(error="boom"),
            forecast=lambda d=7: backend.Outcome(error="boom"),
            operational_report=lambda d=1: backend.Outcome(error="boom"),
            model_cards=lambda: backend.Outcome(error="boom"),
        )
        ctx = ctx_mod.build_context("status", backend=broken)
        self.assertNotIn("today", ctx)
        self.assertGreaterEqual(len(ctx["unavailable"]), 4)
        text, _ = faq.faq_answer("How many beds are available?", ctx)
        self.assertIn("could not be loaded", text)


class TestFaqFallback(unittest.TestCase):
    def test_answers_use_live_numbers(self):
        ctx = ctx_mod.build_context("")
        beds = ctx["today"]["beds"]
        text, matched = faq.faq_answer("How many beds are available?", ctx)
        self.assertTrue(matched)
        self.assertIn(str(beds["available"]), text)

    def test_unmatched_question_says_so_and_is_labelled(self):
        text, matched = faq.faq_answer("Tell me a joke", {})
        self.assertFalse(matched)
        self.assertIn("not generative AI", text)

    def test_alert_answer_matches_active_count(self):
        ctx = {"alerts": {"active_count": 0, "alerts": []}}
        text, _ = faq.faq_answer("Why are there no active alerts?", ctx)
        self.assertIn("No alert is active", text)


class TestMissingApiKey(unittest.TestCase):
    def test_no_secrets_means_not_configured(self):
        self.assertIsNone(llm.config_from_secrets({}))
        self.assertIsNone(llm.config_from_secrets({"openai": {"api_key": "  "}}))

    def test_key_formats(self):
        self.assertEqual(llm.config_from_secrets({"openai": {"api_key": "k", "model": "m"}}), llm.LLMConfig("k", "m"))
        self.assertEqual(llm.config_from_secrets({"OPENAI_API_KEY": "k"}).model, llm.DEFAULT_MODEL)

    def test_key_is_never_in_repr(self):
        self.assertNotIn("sk-test-secret", repr(CONFIG))

    def test_answer_without_config_uses_faq_and_explains(self):
        post = ok_post()
        reply = service.answer("How many beds are available?", config=None, post=post)
        self.assertEqual(reply.source, service.SOURCE_FAQ)
        self.assertIn("not configured", reply.notice)
        self.assertEqual(post.calls, [])  # no network call without a key


class TestApiErrors(unittest.TestCase):
    def check(self, status, expect):
        post = lambda *a, **k: fake_response(status, {})
        reply = service.answer("How many beds are available?", config=CONFIG, post=post)
        self.assertEqual(reply.source, service.SOURCE_FAQ)
        self.assertIn(expect, reply.notice)
        self.assertNotIn("sk-test-secret", reply.notice + reply.text)

    def test_http_errors_degrade_to_faq(self):
        self.check(401, "rejected")
        self.check(429, "rate-limited")
        self.check(404, "not found")
        self.check(500, "HTTP 500")

    def test_timeout_and_connection_errors(self):
        for exc, expect in ((requests.Timeout(), "too long"), (requests.ConnectionError(), "Could not reach")):
            def post(*a, **k):
                raise exc
            reply = service.answer("How many beds are available?", config=CONFIG, post=post)
            self.assertEqual(reply.source, service.SOURCE_FAQ)
            self.assertIn(expect, reply.notice)

    def test_malformed_response(self):
        post = lambda *a, **k: fake_response(200, {"unexpected": True})
        reply = service.answer("How many beds are available?", config=CONFIG, post=post)
        self.assertIn("unexpected response", reply.notice)

    def test_unexpected_exception_does_not_crash(self):
        def post(*a, **k):
            raise RuntimeError("anything")
        reply = service.answer("How many beds are available?", config=CONFIG, post=post)
        self.assertEqual(reply.source, service.SOURCE_FAQ)


class TestLlmRequest(unittest.TestCase):
    def test_request_carries_live_context_history_and_guidance(self):
        post = ok_post("Answer")
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
        reply = service.answer("How many beds are available?", history, config=CONFIG, post=post)
        self.assertEqual((reply.source, reply.text, reply.notice), (service.SOURCE_AI, "Answer", ""))
        call = post.calls[0]
        self.assertEqual(call["json"]["model"], "test-model")
        self.assertEqual(call["headers"]["Authorization"], "Bearer sk-test-secret")
        msgs = call["json"]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        system = msgs[0]["content"]
        state = backend.current_state().value
        self.assertIn(state.as_of_date, system)
        self.assertIn(str(state.available_beds), system)
        self.assertNotIn("sk-test-secret", json.dumps(msgs))

    def test_history_is_limited(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(40)]
        post = ok_post()
        service.answer("status?", history, config=CONFIG, post=post)
        self.assertEqual(len(post.calls[0]["json"]["messages"]), 1 + service.HISTORY_TURNS + 1)


class TestGuardrails(unittest.TestCase):
    def test_clinical_requests_are_declined_without_calling_the_model(self):
        post = ok_post()
        for q in ("Can you diagnose my chest pain?", "What medication for fever should I take?",
                  "Prescribe something for this patient", "What treatment does he need?"):
            reply = service.answer(q, config=CONFIG, post=post)
            self.assertEqual(reply.source, service.SOURCE_GUARD, q)
            self.assertIn("can't help with diagnosis", reply.text)
        self.assertEqual(post.calls, [])

    def test_operational_questions_are_not_blocked(self):
        for q in prompts.SUGGESTED_QUESTIONS:
            self.assertFalse(guardrails.is_clinical_request(q), q)

    def test_empty_and_long_input(self):
        self.assertEqual(service.answer("   ", config=CONFIG, post=ok_post()).source, service.SOURCE_GUARD)
        self.assertLessEqual(len(guardrails.clean_question("x " * 2000)), guardrails.MAX_QUESTION_CHARS)

    def test_system_prompt_contains_required_rules(self):
        p = prompts.build_system_prompt({"x": 1}).lower()
        for needle in ("never invent", "recorded", "expected", "estimated", "hypothetical", "synthetic",
                       "diagnosis", "not medical advice", "hinglish"):
            self.assertIn(needle, p)


class TestPage(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        from tests.test_navigation_rail import APP

        at = AppTest.from_file(APP, default_timeout=240)
        at.query_params["view"] = "assistant"
        at.session_state[auth.SESSION_KEY] = auth.DEMO_USER
        at.run()
        return at

    def test_navigation_item_exists(self):
        self.assertIn("assistant", VIEW_IDS)
        self.assertIn(("Ask", ["assistant"]), NAV_GROUPS)

    def test_page_renders_welcome_and_suggestions_and_answers_in_faq_mode(self):
        at = self._app()
        self.assertFalse(at.exception)
        self.assertEqual(len([b for b in at.button if str(b.key).startswith("sq-")]), 12)
        at.button(key="sq-1").click().run()
        self.assertFalse(at.exception)
        names = [m.name for m in at.chat_message]
        self.assertEqual(names, ["assistant", "user", "assistant"])
        self.assertIn("beds available", " ".join(m.value for m in at.chat_message[2].markdown))
        at.button(key="assistant-clear").click().run()
        self.assertEqual([m.name for m in at.chat_message], ["assistant"])


if __name__ == "__main__":
    unittest.main()
