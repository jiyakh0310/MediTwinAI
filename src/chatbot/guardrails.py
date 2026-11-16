"""Input guardrails: clinical requests are declined before any model call."""

from __future__ import annotations

import re

MAX_QUESTION_CHARS = 600

_CLINICAL = re.compile(
    r"\b(diagnos\w*|prescri\w+|treat(?:ment|ing)?|dosage|dose|medication for|what (?:medicine|drug|tablet)s? "
    r"(?:should|to) (?:i|he|she|they)|should i take|do i have|is it (?:cancer|diabetes|covid)|cure for|"
    r"triage (?:this|my|a) patient|my symptoms|his symptoms|her symptoms)\b",
    re.IGNORECASE,
)

CLINICAL_REFUSAL = (
    "I can't help with diagnosis, treatment, prescriptions or clinical triage. MediTwin AI is an educational "
    "hospital-operations project on synthetic data, and nothing here is medical advice. For a health concern, "
    "please speak to a qualified clinician. I can explain beds, ICU, outpatients, waiting time, medicine "
    "stock, staff workload, alerts, the Digital Twin, What-If and the ML models."
)

EMPTY_REPLY = "Please type a question about the hospital's operations or the MediTwin models."


def clean_question(text: str | None) -> str:
    return " ".join((text or "").split())[:MAX_QUESTION_CHARS]


def is_clinical_request(text: str) -> bool:
    return bool(_CLINICAL.search(text or ""))
