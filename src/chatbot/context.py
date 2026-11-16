"""Live context for the assistant, taken from the existing backend functions.

Nothing here recalculates a metric: every number comes from ``src.ui.backend`` (cached twin,
forecaster, alert engine, simulator) or the saved ``metrics.json`` files. Each value carries its
provenance (Recorded / Expected / Estimated / Simulated). Anything unavailable is listed, never guessed.
"""

from __future__ import annotations

import re
from typing import Any

from src.alerts.models import AlertThresholds
from src.chatbot import knowledge

_WHATIF_WORDS = re.compile(r"what[\s-]*if|scenario|simulat|inflow|surge|increase|decrease|delay|\bif\b", re.I)

REC, EXP, EST, SIM = "Recorded", "Expected (ML prediction)", "Estimated (rule-based)", "Simulated (hypothetical)"


def wants_whatif(question: str) -> bool:
    return bool(_WHATIF_WORDS.search(question or ""))


def _backend():
    from src.ui import backend

    return backend


def _round(value: Any, digits: int = 1) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _state_block(state: Any, label: str) -> dict[str, Any]:
    from src.ui import data as ui_data

    pressure = ui_data.state_staff_pressure(state)
    use = ui_data.estimated_medicine_consumption(state)
    cover = (state.medicine_stock / use) if use and use > 0 else None
    return {
        "label": label, "date": state.as_of_date, "capacity_status": state.capacity_status,
        "beds": {"occupied": state.occupied_beds, "total": state.total_beds, "available": state.available_beds,
                 "occupancy_percent": _round(state.bed_occupancy_percentage)},
        "icu": {"occupied": state.occupied_icu_beds, "total": state.total_icu_beds,
                "available": state.total_icu_beds - state.occupied_icu_beds,
                "occupancy_percent": _round(state.icu_occupancy_percentage)},
        "outpatients_opd": state.current_opd, "emergency_cases": state.emergency_cases,
        "admissions": state.admissions, "doctors": state.doctors_available, "nurses": state.nurses_available,
        "waiting_time_minutes": _round(state.average_waiting_time),
        "medicine_stock_units": state.medicine_stock,
        "estimated_daily_medicine_use": _round(use),
        "medicine_days_of_cover": _round(cover),
        "staff_workload_patients_per_staff": _round(pressure, 2),
    }


def _model_summary(cards: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pr = (cards.get("patient_risk") or {}).get("selected") or {}
    if pr:
        out["patient_risk"] = {"model": pr.get("model"), "accuracy": _round(pr.get("accuracy"), 3),
                               "macro_f1": _round(pr.get("f1_macro"), 3),
                               "high_risk_recall": _round(pr.get("recall_high"), 3), "evaluated_on": "held-out test split"}
    wt = (cards.get("waiting_time") or {}).get("selected") or {}
    if wt:
        out["waiting_time"] = {"model": wt.get("model"), "mae_minutes": _round(wt.get("mae"), 2),
                               "rmse_minutes": _round(wt.get("rmse"), 2), "r2": _round(wt.get("r2"), 3),
                               "evaluated_on": "chronological test split"}
    opd = cards.get("opd_forecast") or {}
    if opd:
        hold, itv = opd.get("metrics_holdout") or {}, opd.get("interval") or {}
        out["opd_forecast"] = {"model": opd.get("selected_model"), "test_mae": _round(hold.get("mae"), 2),
                               "test_rmse": _round(hold.get("rmse"), 2), "test_r2": _round(hold.get("r2"), 3),
                               "interval_halfwidth": _round(itv.get("halfwidth"), 2),
                               "interval_test_coverage": itv.get("coverage_test"),
                               "note": "Compare with the weekday-average baseline in the Forecasts page."}
    an = cards.get("anomaly") or {}
    if an:
        out["anomaly_detection"] = {"model": an.get("model"), "contamination": an.get("contamination"),
                                    "days_flagged_in_sample": an.get("n_anomalies_flagged")}
    return out


def _thresholds() -> dict[str, Any]:
    t = AlertThresholds()
    return {"bed_and_icu_occupancy_percent_low_medium_high_critical": t.bed_occupancy,
            "opd_zscore": t.opd_zscore, "staff_pressure_zscore": t.staff_pressure_zscore,
            "medicine_days_cover_below": t.medicine_days_cover,
            "rising_opd_percent": t.rising_opd_percent, "anomaly_high_score": t.anomaly_high_score}


def _whatif_block(backend: Any, date_iso: str, unavailable: list[str]) -> list[dict[str, Any]]:
    from src.ui import data as ui_data

    base = backend.baseline_state(date_iso)
    if not base.ok:
        unavailable.append(f"What-If baseline: {base.error}")
        return []
    results = []
    for name in ui_data.PRESETS:
        controls = ui_data.apply_preset(base.value, name)
        res = backend.run_whatif(date_iso, controls)
        if not res.ok:
            unavailable.append(f"What-If '{name}': {res.error}")
            continue
        sim = res.value.simulated_state
        results.append({
            "scenario": name, "type": SIM, "capacity_status": sim.capacity_status,
            "beds_occupied": sim.occupied_beds, "beds_total": sim.total_beds,
            "bed_occupancy_percent": _round(sim.bed_occupancy_percentage),
            "icu_occupied": sim.occupied_icu_beds, "icu_total": sim.total_icu_beds,
            "icu_occupancy_percent": _round(sim.icu_occupancy_percentage),
            "outpatients_opd": sim.current_opd, "waiting_time_minutes": _round(sim.average_waiting_time),
            "medicine_stock_units": sim.medicine_stock,
            "bed_shortfall": res.value.bed_shortfall, "icu_shortfall": res.value.icu_shortfall,
        })
    return results


def build_context(question: str = "", backend: Any | None = None) -> dict[str, Any]:
    """Collect live, labelled values. Failed lookups go to ``unavailable``."""
    b = backend or _backend()
    unavailable: list[str] = []
    ctx: dict[str, Any] = {"project": knowledge.PROJECT_NOTE, "value_types": knowledge.VALUE_TYPES}

    def need(label: str, outcome: Any) -> Any:
        if not outcome.ok:
            unavailable.append(f"{label}: {outcome.error}")
            return None
        return outcome.value

    cur = need("Current state", b.current_state())
    if cur is not None:
        ctx["data_through"] = cur.as_of_date
        ctx["today"] = _state_block(cur, "Latest day in the dataset")
        mw = need("Model waiting time", b.current_model_wait())
        if mw is not None:
            ctx["today"]["waiting_time_model_estimate_minutes"] = {"value": _round(mw), "type": EXP}
        ctx["today"]["field_types"] = {
            "beds, icu, outpatients_opd, emergency_cases, admissions, doctors, nurses, medicine_stock_units, "
            "waiting_time_minutes": REC,
            "estimated_daily_medicine_use, medicine_days_of_cover, staff_workload_patients_per_staff": EST}
    nxt = need("Next-day state", b.predicted_state(1))
    if nxt is not None:
        ctx["tomorrow"] = {"types": "outpatients_opd and waiting_time_minutes are Expected (ML); the rest are "
                                    "Estimated (rule-based projections)", **_state_block(nxt, "Next day")}
    fc = need("OPD forecast", b.forecast(7))
    if fc is not None:
        ctx["opd_forecast_7_days"] = {"type": EXP, "days": [
            {"date": str(r["date"])[:10], "expected_opd": _round(r["opd_forecast"], 0),
             "range_low": _round(r["yhat_lower"], 0), "range_high": _round(r["yhat_upper"], 0)}
            for r in fc.to_dict("records")]}
    rep = need("Alerts", b.operational_report(1))
    if rep is not None:
        report, recs = rep
        ctx["alerts"] = {
            "type": "Rule-based thresholds (the anomaly alert uses the Isolation Forest)",
            "active_count": len(report.alerts), "evaluated_for": report.generated_for,
            "alerts": [{"severity": a.severity.label, "title": a.title, "scope": a.scope, "basis": a.basis,
                        "explanation": a.explanation} for a in report.alerts],
            "checks_not_performed": list(report.unavailable)}
        ctx["recommendations"] = [{"severity": r.severity.label, "title": r.title, "action": r.action,
                                   "why": r.rationale} for r in recs]
    ctx["alert_thresholds"] = _thresholds()
    cards = need("Model metrics", b.model_cards())
    if cards is not None:
        ctx["model_metrics"] = _model_summary(cards)
    ctx["ml_tasks"] = knowledge.ML_TASKS
    ctx["how_it_works"] = knowledge.HOW_IT_WORKS
    if cur is not None and wants_whatif(question):
        ctx["what_if_presets"] = {"note": "Hypothetical results on the latest day; not forecasts.",
                                  "results": _whatif_block(b, cur.as_of_date, unavailable)}
    ctx["unavailable"] = unavailable
    return ctx
