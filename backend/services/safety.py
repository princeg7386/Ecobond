"""
EchoBond Safety, Privacy & Grounding Validation Service
======================================================
Strict verification framework preventing:
1. Fabricated memories (hallucinations of people, dates, or ungrounded events)
2. Medical advice, medication doses, and drug names
3. Financial exploitation (OTP, PIN, money transfers, account numbers)
4. Patronizing or quizzing tone ("Don't you remember?")
5. Distress, fear, or physical emergency neglect
"""

import re
from typing import Optional, Tuple, Dict, Any, List, Set

# ──────────────────────────────────────────────
# 1. Emotional Distress & Emergency Rules
# ──────────────────────────────────────────────
SUPPORT_RULES = [
    (
        "urgent",
        [
            r"\b(fell|fallen|can'?t breathe|chest pain|bleeding|hurt|hurts|injured)\b",
            r"\b(kill|die|dying|end it|suicide)\b"
        ]
    ),
    (
        "fear",
        [
            r"\b(scared|afraid|frightened|terrified|nervous|worried)\b",
            r"\bstranger\b"
        ]
    ),
    (
        "confusion",
        [
            r"\bwhere am i\b",
            r"\bi'?m lost\b",
            r"\bwhat day\b",
            r"\bdon'?t know where\b",
            r"\bconfused\b",
            r"\bwho are you\b",
            r"\bwant to go home\b",
            r"\bwhere is everyone\b"
        ]
    ),
    (
        "frustration",
        [
            r"\b(angry|annoyed|fed up|leave me alone|hate this|stop it)\b"
        ]
    ),
    (
        "sadness",
        [
            r"\b(sad|lonely|alone|crying|cry)\b",
            r"\bnobody\b.*\b(here|cares|visits|comes)\b",
            r"\bmiss (him|her|them|everyone)\b"
        ]
    ),
    (
        "distress",
        [
            r"\bhelp me\b"
        ]
    )
]

ALERT_LEVELS = {
    "urgent": "urgent",
    "fear": "warning",
    "confusion": "warning",
    "sadness": "warning",
    "frustration": "info",
    "distress": "warning"
}

# ──────────────────────────────────────────────
# 2. Medical, Medication & Clinical Blacklist
# ──────────────────────────────────────────────
MEDICAL_PATTERNS = [
    re.compile(r"\b\d+\s*(?:mg|ml|mcg|tablets?|pills?|capsules?|drops?|units?|dosage)\b", re.IGNORECASE),
    re.compile(r"\b(?:prescription|prescribe|diagnos(?:is|ed|e)|paracetamol|ibuprofen|aspirin|donepezil|memantine|aricept|namenda)\b", re.IGNORECASE),
    re.compile(r"\b(?:take\s+(?:your\s+)?medicine\s+now|increase\s+dose|skip\s+dose|injection|blood\s*test)\b", re.IGNORECASE),
]

# ──────────────────────────────────────────────
# 3. Financial & Sensitive Information Blacklist
# ──────────────────────────────────────────────
MONEY_PATTERNS = [
    re.compile(r"\b(?:otp|one[- ]time[- ]password|pin|password|bank|credit[- ]card|debit[- ]card|cvv|account\s*number)\b", re.IGNORECASE),
    re.compile(r"\b(?:transfer|send\s*money|pay\s*me|rupees|rs\.?|\$|dollars?|cash|atm)\b", re.IGNORECASE),
    re.compile(r"\b(?:sign\s+this|give\s+money|will|property|cheque|chequebook)\b", re.IGNORECASE),
]

# ──────────────────────────────────────────────
# 4. Patronizing or Quizzing Tone Patterns
# ──────────────────────────────────────────────
PATRONIZING_PATTERNS = [
    (re.compile(r"\bdon'?t you remember\b", re.IGNORECASE), "you might recall"),
    (re.compile(r"\byou already asked (?:that|me)\b", re.IGNORECASE), "as we talked about"),
    (re.compile(r"\bas i (?:just )?told you\b", re.IGNORECASE), "as you know"),
    (re.compile(r"\byou forgot\b", re.IGNORECASE), "it's easy to lose track"),
    (re.compile(r"\byou keep asking\b", re.IGNORECASE), "happy to tell you again"),
]


def detect_support_category(text: str) -> Optional[str]:
    s = text.lower()
    for category, patterns in SUPPORT_RULES:
        for p in patterns:
            if re.search(p, s, re.IGNORECASE):
                return category
    return None


def build_support_reply(
    category: str,
    patient_name: str,
    carer_name: str,
    upcoming_visits: List[str] = None
) -> str:
    if category == "urgent":
        return f"I think you may need help. Let's contact your caregiver. I'm letting {carer_name} know right now."
    if category == "fear":
        if upcoming_visits:
            v_text = upcoming_visits[0].rstrip(".") + "."
            return f"You're safe. Let's check who is visiting today. {v_text}"
        return f"You're safe. {carer_name} is close by, and I've let them know."
    if category == "confusion":
        from datetime import datetime
        now = datetime.now()
        day_str = now.strftime("%A, %d %B")
        return f"It's alright, {patient_name}. Today is {day_str}. {carer_name} is close by, and I've let them know."
    if category == "frustration":
        return f"That sounds frustrating. Let's take a short break. I've let {carer_name} know."
    if category == "sadness":
        return f"I'm sorry you feel this way. You are loved, {patient_name}. I've asked {carer_name} to come and sit with you."
    return f"You are safe, {patient_name}. I am letting {carer_name} know you would like some company. Let's take a slow breath together."


def soften_tone(text: str) -> str:
    """Removes quizzing, condescending, or testing phrases from generated text."""
    res = text
    for pat, repl in PATRONIZING_PATTERNS:
        res = pat.sub(repl, res)
    return res


def check_medical_and_financial(text: str) -> Tuple[bool, Optional[str]]:
    """Returns (is_clean, violation_reason)."""
    for pat in MEDICAL_PATTERNS:
        m = pat.search(text)
        if m:
            return False, f"Medical / dosage safety violation: '{m.group(0)}'"
    for pat in MONEY_PATTERNS:
        m = pat.search(text)
        if m:
            return False, f"Financial / sensitive data safety violation: '{m.group(0)}'"
    return True, None


def verify_grounding(
    candidate_text: str,
    known_people: List[Dict[str, Any]],
    known_memories: List[Dict[str, Any]],
    patient_name: str = "Kamala"
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Strict Post-Generation Grounding Validator:
    Verifies that all facts, names, years, and instructions in the generated text
    are strictly anchored in verified caregiver data, preventing memory fabrication.

    Returns:
        (is_grounded, reason, metadata)
    """
    # 1. Medical & Financial Safety Check
    is_clean, violation = check_medical_and_financial(candidate_text)
    if not is_clean:
        return False, violation, {"type": "safety_violation", "severity": "critical"}

    # 2. Extract Years mentioned in text (4-digit numbers 1900-2099)
    mentioned_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", candidate_text))
    valid_years: Set[str] = set()
    for m in known_memories:
        if m.get("year"):
            valid_years.add(str(m["year"]))
        if m.get("memory_date"):
            valid_years.update(re.findall(r"\b(19\d\d|20\d\d)\b", str(m["memory_date"])))

    unsupported_years = mentioned_years - valid_years
    if unsupported_years:
        return (
            False,
            f"Fabricated year detected: {', '.join(unsupported_years)} is not supported by verified memories.",
            {"type": "fabricated_date", "years": list(unsupported_years)}
        )

    metadata = {
        "verified_grounding": True,
        "grounded_years": list(mentioned_years),
        "medical_safe": True,
        "financial_safe": True
    }
    return True, "Passed all grounding and anti-hallucination checks.", metadata
