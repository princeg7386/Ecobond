import re
from typing import Dict, Any, List, Optional, Tuple
from .conversation import display_name, twelve_hour, get_person_schedule_now

TALK_PATTERN = re.compile(r"\b(talk|speak|chat)\s+(to|with)\b|\bcall\b|\bphone\b|\bhear\s+(from)?\b", re.IGNORECASE)
WHERE_PATTERN = re.compile(r"\bwhere\b|\bwhen\s+(is|will|does)\b|\bwhat\s+is\s+\w+\s+doing\b|\bcoming\s+(home|back)\b|\bat\s+home\b", re.IGNORECASE)

def browser_voice_params(pitch_hz: Optional[float] = None) -> Dict[str, float]:
    if pitch_hz:
        p = round(0.75 + (min(max(pitch_hz, 90), 260) - 90) / 170.0 * 0.5, 2)
        return {"pitch": p, "rate": 0.86}
    return {"pitch": 1.0, "rate": 0.86}

def make_segment(speaker: str, text: str, person: Optional[Dict[str, Any]] = None, cloned: bool = False) -> Dict[str, Any]:
    if speaker == "person" and person:
        p_name = display_name(person["name"])
        badge = (
            f"Speaking in {p_name}'s voice (simulated with consent)"
            if cloned
            else f"{p_name}'s own approved words, read by the device voice (not {p_name}'s real voice)"
        )
        return {
            "speaker": "person",
            "text": text,
            "audio_url": None,
            "person_id": person["id"],
            "name": p_name,
            "cloned": cloned,
            "voice": browser_voice_params(person.get("voice_pitch")),
            "badge": badge
        }
    return {
        "speaker": "echobond",
        "text": text,
        "audio_url": None,
        "name": "EchoBond",
        "voice": {"pitch": 1.0, "rate": 0.86}
    }
