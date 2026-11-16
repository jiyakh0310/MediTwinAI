"""Limited keyword FAQ used when no API key is configured (or the AI service fails).

This is NOT generative AI: it matches keywords to fixed answer templates and fills them with the live
values already collected by ``context.build_context``. It says so in every reply.
"""

from __future__ import annotations

from typing import Any, Callable

from src.chatbot import knowledge

FAQ_LABEL = "_Answered by the built-in FAQ (keyword matching, not generative AI)._"


def _fmt(value: Any, unit: str = "") -> str:
    return "not available" if value is None else f"{value}{unit}"


def _today(ctx: dict) -> dict | None:
    return ctx.get("today")


def _need(ctx: dict, key: str) -> str | None:
    return None if ctx.get(key) else "That value could not be loaded right now, so I can't answer it."


def _status(ctx: dict) -> str:
    t = _today(ctx)
    if not t:
        return _need(ctx, "today")
    n = (ctx.get("alerts") or {}).get("active_count")
    return (
        f"Latest day in the dataset: **{t['date']}** (synthetic data). Capacity status: **{t['capacity_status']}**.\n"
        f"- Beds: {t['beds']['occupied']}/{t['beds']['total']} used ({t['beds']['occupancy_percent']}%) - Recorded\n"
        f"- ICU: {t['icu']['occupied']}/{t['icu']['total']} used ({t['icu']['occupancy_percent']}%) - Recorded\n"
        f"- Outpatients: {t['outpatients_opd']} - Recorded\n"
        f"- Active alerts: {_fmt(n)}"
    )


def _beds(ctx: dict) -> str:
    t = _today(ctx)
    if not t:
        return _need(ctx, "today")
    b = t["beds"]
    return (f"On {t['date']}: **{b['available']} beds available** out of {b['total']} "
            f"({b['occupied']} occupied, {b['occupancy_percent']}%). These are Recorded values.")


def _icu(ctx: dict) -> str:
    t = _today(ctx)
    if not t:
        return _need(ctx, "today")
    i = t["icu"]
    return (f"On {t['date']}: ICU occupancy is **{i['occupancy_percent']}%** ({i['occupied']}/{i['total']}), "
            f"so {i['available']} ICU beds are free. Recorded value.")


def _tomorrow_opd(ctx: dict) -> str:
    f = (ctx.get("opd_forecast_7_days") or {}).get("days")
    if not f:
        return "The OPD forecast could not be loaded right now, so I can't answer it."
    d = f[0]
    return (f"Expected outpatients for **{d['date']}**: about **{d['expected_opd']:.0f}** "
            f"(range {d['range_low']:.0f}-{d['range_high']:.0f}). This is an Expected value from the "
            "XGBoost forecasting model.")


def _wait_how(ctx: dict) -> str:
    return knowledge.HOW_IT_WORKS["waiting_time"]


def _medicine(ctx: dict) -> str:
    t = _today(ctx)
    if not t:
        return _need(ctx, "today")
    return (f"Stock is **{t['medicine_stock_units']:,} units** (Recorded). At an estimated "
            f"{_fmt(t['estimated_daily_medicine_use'])} units/day, that is about "
            f"**{_fmt(t['medicine_days_of_cover'])} days of cover** (Estimated, rule-based).\n\n"
            + knowledge.HOW_IT_WORKS["medicine"])


def _staff(ctx: dict) -> str:
    t = _today(ctx)
    if not t:
        return _need(ctx, "today")
    return (f"Workload is **{_fmt(t['staff_workload_patients_per_staff'])} patients per clinical staff member** "
            f"({t['doctors']} doctors, {t['nurses']} nurses). Estimated, rule-based.\n\n"
            + knowledge.HOW_IT_WORKS["staff"])


def _alerts(ctx: dict) -> str:
    a = ctx.get("alerts")
    if not a:
        return "Alerts could not be loaded right now, so I can't answer it."
    if a["active_count"] == 0:
        return ("No alert is active because no rule threshold has been crossed for the current day or the next "
                "day. " + knowledge.HOW_IT_WORKS["alerts"])
    lines = [f"- **{x['severity']}**: {x['title']}" for x in a["alerts"]]
    return f"There are **{a['active_count']} active alert(s)**:\n" + "\n".join(lines)


def _bed_90(ctx: dict) -> str:
    low, med, high, crit = (ctx.get("alert_thresholds") or {}).get(
        "bed_and_icu_occupancy_percent_low_medium_high_critical", (None,) * 4)
    if low is None:
        return "The alert thresholds could not be loaded right now."
    return (f"Bed alerts are rule-based: Low at {low:g}%, Medium at {med:g}%, **High at {high:g}%**, Critical at "
            f"{crit:g}%. So crossing 90% raises a High bed alert (and recommendations are linked to it). "
            "Separately, the capacity status turns Moderate/High/Critical from its own cut-offs.")


def _opd_explain(ctx: dict) -> str:
    return knowledge.HOW_IT_WORKS["opd_forecast"]


def _models(ctx: dict) -> str:
    return "\n".join(f"- **{m['task']}**: {m['model']}. {m['limit']}" for m in knowledge.ML_TASKS)


def _inflow(ctx: dict) -> str:
    rows = (ctx.get("what_if_presets") or {}).get("results") or []
    row = next((r for r in rows if r["scenario"] == "Inflow +20%"), None)
    if not row:
        return knowledge.HOW_IT_WORKS["whatif"]
    return (f"**Hypothetical** (What-If, not a forecast): with patient inflow +20%, outpatients would be "
            f"{row['outpatients_opd']}, beds {row['beds_occupied']}/{row['beds_total']} "
            f"({row['bed_occupancy_percent']}%), ICU {row['icu_occupied']}/{row['icu_total']}, waiting time "
            f"about {row['waiting_time_minutes']} min, status **{row['capacity_status']}**.")


def _synthetic(ctx: dict) -> str:
    return knowledge.PROJECT_NOTE + " " + knowledge.VALUE_TYPES


# (keyword groups: every group needs at least one hit), answer
_FAQ: list[tuple[list[tuple[str, ...]], Callable[[dict], str]]] = [
    ([("hospital status", "overall status", "current status", "overview", "how is the hospital")], _status),
    ([("bed",), ("available", "free", "vacant", "how many")], _beds),
    ([("icu",), ("occupancy", "current", "how many", "status", "free")], _icu),
    ([("opd", "outpatient"), ("explain", "how", "work")], _opd_explain),
    ([("opd", "outpatient"), ("tomorrow", "next day", "expected", "forecast")], _tomorrow_opd),
    ([("waiting",), ("calculat", "how", "predict", "work")], _wait_how),
    ([("medicine", "stock"), ("days", "remain", "cover", "left")], _medicine),
    ([("staff", "workload", "doctor", "nurse")], _staff),
    ([("alert",), ("why", "no", "active", "any")], _alerts),
    ([("90",), ("bed", "occupancy")], _bed_90),
    ([("model", "ml", "machine learning", "algorithm")], _models),
    ([("inflow", "20%", "surge", "what if", "what-if", "scenario")], _inflow),
    ([("synthetic", "real hospital", "dataset", "data")], _synthetic),
]


def match(question: str) -> Callable[[dict], str] | None:
    q = (question or "").lower()
    for groups, answer in _FAQ:
        if all(any(k in q for k in group) for group in groups):
            return answer
    return None


FAQ_MISS = (
    "I can only answer a fixed set of questions without the AI service. Try one of the suggested questions, "
    "for example about beds, ICU, tomorrow's OPD, medicine cover, staff workload, alerts or the ML models."
)


def faq_answer(question: str, ctx: dict[str, Any]) -> tuple[str, bool]:
    """(text, matched). The text always carries the 'not generative AI' label."""
    fn = match(question)
    if fn is None:
        return f"{FAQ_MISS}\n\n{FAQ_LABEL}", False
    return f"{fn(ctx)}\n\n{FAQ_LABEL}", True
