"""Small static knowledge base: what the project is and how each part works.

Contains no live hospital figures. Live numbers come from ``context.py``.
"""

from __future__ import annotations

PROJECT_NOTE = (
    "MediTwin AI is an educational hospital digital twin. All data is synthetic (366 daily operations "
    "rows and 1,000 patient records) and the app is not connected to a real hospital."
)

VALUE_TYPES = (
    "Recorded = a value stored in the dataset. Expected = a prediction from a trained ML model. "
    "Estimated = a value from a fixed rule/formula. Simulated = a hypothetical What-If result."
)

ML_TASKS = [
    {"task": "Patient risk (Low/Medium/High)", "model": "XGBoost classifier",
     "data": "1,000 synthetic patient records (vitals, age, history)",
     "limit": "Labels come from a rule-based score, so the model learns that rule; High-risk recall is modest."},
    {"task": "Waiting-time prediction (minutes)", "model": "Huber regressor pipeline",
     "data": "Daily operations features, chronological train/validation/test split",
     "limit": "Trained on one synthetic year; it estimates a daily average, not an individual wait."},
    {"task": "OPD (outpatient) forecast", "model": "XGBoost forecaster on calendar + lag features",
     "data": "Daily outpatient counts; beyond 7 days it feeds its own forecasts back in (recursive)",
     "limit": "Accuracy is close to a simple weekday-average baseline; the interval is a residual-based range."},
    {"task": "Anomaly detection (unusual days)", "model": "Isolation Forest",
     "data": "Standardised daily operations values",
     "limit": "Few labelled spikes in synthetic data, so the evaluation is small."},
]

HOW_IT_WORKS = {
    "waiting_time": (
        "Waiting time is predicted by a trained Huber regression pipeline from daily operational features "
        "(patient load, staffing, occupancy, calendar and recent history). It is an Expected value. "
        "The dataset's own waiting time for a past day is Recorded."
    ),
    "opd_forecast": (
        "The OPD forecast is an XGBoost model using day-of-week, weekend/holiday flags, month (sin/cos) and "
        "outpatient counts from 7/14/21/28 days earlier. Days 1-7 use real history; longer horizons are "
        "recursive. The shaded range is the forecast plus or minus a residual-based interval."
    ),
    "twin": (
        "The Digital Twin shows today's recorded state beside a predicted state. Outpatients and waiting time "
        "come from ML models; beds, ICU, emergency cases, medicine and staff are rule-based projections."
    ),
    "whatif": (
        "What-If changes conditions (patient inflow, emergencies, beds, ICU, staff, medicine stock) and "
        "recomputes the day with fixed rules plus the waiting-time model. Results are hypothetical, never "
        "forecasts of what will actually happen."
    ),
    "alerts": (
        "Alerts are rule-based threshold checks (occupancy, outpatient and staff z-scores, medicine cover, "
        "rising outpatients) plus one anomaly check from the Isolation Forest. Recommendations are fixed "
        "action text linked to alerts."
    ),
    "staff": (
        "Staff workload is (outpatients + emergency cases + admissions) divided by (doctors + nurses): "
        "patients per clinical staff member. It is an Estimated (rule-based) value."
    ),
    "medicine": (
        "Medicine days of cover = stock divided by estimated daily use. Estimated daily use is a fixed "
        "formula based on outpatients, emergency cases and admissions, so it is Estimated, not Recorded."
    ),
}
