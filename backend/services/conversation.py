import re
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

from .safety import detect_support_category, build_support_reply, ALERT_LEVELS, verify_grounding, soften_tone
from .memory_store import search_memories, tokenize, normalize_question

FOLLOW_PATTERN = re.compile(r"\b(we|us|together|they|them|he|she|him|her|his|there|else|that|then|more)\b", re.IGNORECASE)
WHO_ELSE_PATTERN = re.compile(r"\bwho else\b|\bwho (was|were) (there|with (us|me|them))\b|\banyone else\b", re.IGNORECASE)
WHO_IS_PATTERN = re.compile(r"^\s*(who\s+is|who's|who\s+was)\b", re.IGNORECASE)
WHERE_IS_PATTERN = re.compile(r"^\s*(where\s+is|where's|where\s+has)\b|\bwhen (is|will) \w+ (com|visit)", re.IGNORECASE)

def display_name(full_name: str) -> str:
    parts = full_name.split()
    if re.match(r"^dr\.?$", parts[0], re.IGNORECASE):
        return f"Dr. {parts[-1]}"
    return parts[0]

def clean_rel(rel: str) -> str:
    return rel.replace(" & ", " and ")

def twelve_hour(hhmm: str) -> str:
    try:
        parts = str(hhmm or "").split(":")
        h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        ap = "PM" if h >= 12 else "AM"
        hh = h % 12 or 12
        return f"{hh}:{m:02d} {ap}" if m else f"{hh} {ap}"
    except Exception:
        return ""

def day_matches(days_str: str, weekday: int) -> bool:
    d = (days_str or "daily").lower().replace(" ", "")
    if d == "daily":
        return True
    if d == "weekdays":
        return weekday < 5
    if d == "weekends":
        return weekday >= 5
    day_keys = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    return any(day_keys.get(x[:3]) == weekday for x in d.split(","))

def get_person_schedule_now(person_id: int, schedules: List[Dict[str, Any]], at: datetime = None) -> Optional[Dict[str, Any]]:
    if at is None:
        at = datetime.now()
    hhmm = at.strftime("%H:%M")
    wd = at.weekday()  # Monday is 0
    person_scheds = [s for s in schedules if s.get("person_id") == person_id]
    person_scheds.sort(key=lambda x: x.get("start_at", ""))
    for s in person_scheds:
        if day_matches(s.get("days", "daily"), wd) and s.get("start_at", "") <= hhmm < s.get("end_at", ""):
            return s
    return None

class ConversationEngine:
    def __init__(self, db_conn, patient_name: str = "Kamala"):
        self.conn = db_conn
        self.patient = patient_name

    def get_carer_name(self, people: List[Dict[str, Any]]) -> str:
        carer = next((p for p in people if "caregiver" in p.get("relationship", "").lower() and p.get("status") == "active"), None)
        if not carer:
            carer = next((p for p in people if p.get("status") == "active"), None)
        return display_name(carer["name"]) if carer else "your family"

    def answer_query(
        self,
        question: str,
        ctx: Dict[str, Any],
        people: List[Dict[str, Any]],
        memories: List[Dict[str, Any]],
        places: List[Dict[str, Any]],
        reminders: List[Dict[str, Any]],
        schedules: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
        people_map = {p["id"]: p for p in people}
        places_map = {pl["id"]: pl for pl in places}
        carer = self.get_carer_name(people)

        # 1. Safety and distress check
        cat = detect_support_category(question)
        if cat:
            now_str = datetime.now().strftime("%H:%M")
            upcoming_visits = [
                r["text"] for r in reminders
                if r.get("active") and r.get("kind") == "visit" and r.get("time_of_day", "") >= now_str
            ]
            upcoming_visits.sort()
            reply_text = build_support_reply(cat, self.patient, carer, upcoming_visits)
            alert = {
                "level": ALERT_LEVELS.get(cat, "warning"),
                "kind": "distress",
                "reason": "Possible distress in conversation",
                "detail": question,
                "ref": cat
            }
            res = {
                "text": reply_text,
                "engine": "safety",
                "sources": [],
                "why": [{
                    "type": "rule",
                    "text": f"Words in the question matched EchoBond's '{cat}' support rule, so it gave a calm, fixed reply and let the caregiver know"
                }],
                "confidence": "None",
                "category": cat,
                "alert": ALERT_LEVELS.get(cat, "warning"),
                "people": [],
                "memory_ids": [],
                "outcome": "safety"
            }
            return res, alert, ctx

        # 2. Protected persons check
        low = question.lower()
        for p in people:
            if p.get("status") != "active":
                first = p["name"].split()[0].lower()
                nick = (p.get("nickname") or "").lower()
                if re.search(rf"\b{re.escape(first)}\b", low) or (nick and re.search(rf"\b{re.escape(nick)}\b", low)):
                    first_disp = p["name"].split()[0]
                    alert = {
                        "level": "warning",
                        "kind": "protected_person",
                        "reason": "Asked about a protected person",
                        "detail": question,
                        "ref": ""
                    }
                    res = {
                        "text": f"{first_disp} is very dear to you. {carer} would love to sit with you and talk about them.",
                        "engine": "safety",
                        "sources": [],
                        "why": [{
                            "type": "rule",
                            "text": "This person is protected by the family, so EchoBond doesn't share memories about them and asked the caregiver to help"
                        }],
                        "confidence": "None",
                        "category": "protected_person",
                        "alert": "warning",
                        "people": [],
                        "memory_ids": [],
                        "outcome": "safety"
                    }
                    return res, alert, ctx

        # 3. Target extraction
        active_people = [p for p in people if p.get("status") == "active"]
        qt = set(tokenize(question))
        named = []
        for p in active_people:
            names = [t.lower().replace(".", "") for t in p["name"].split() if not re.match(r"^dr\.?$", t, re.IGNORECASE)]
            rel_tokens = [t for t in tokenize(p["relationship"].replace("&", " ").replace("-", " ")) if t not in ("friend", "family", "law")]
            if any(t in qt for t in names) or (p.get("nickname") and p["nickname"].lower() in qt):
                named.append(p["id"])
            elif rel_tokens and all(t in qt or t == "caregiver" for t in rel_tokens) and any(t in qt for t in rel_tokens):
                named.append(p["id"])

        follow = not named and bool(FOLLOW_PATTERN.search(question)) and bool(ctx.get("person_ids") or ctx.get("memory_ids"))
        person_ids = named if named else (ctx.get("person_ids", []) if follow else [])
        exclude_ids = set(ctx.get("told_ids", [])) if follow else set()

        # 4. WHO ELSE rule
        if WHO_ELSE_PATTERN.search(question):
            mem_ids = ctx.get("memory_ids", [])
            if not mem_ids:
                return {
                    "text": "Which memory do you mean? You can ask me about a person or a place.",
                    "engine": "template",
                    "sources": [],
                    "why": [{"type": "rule", "text": "There was no recent memory in this conversation to refer back to"}],
                    "confidence": "None",
                    "category": None,
                    "alert": None,
                    "people": [],
                    "memory_ids": [],
                    "outcome": "no_memory"
                }, None, ctx

            m = next((x for x in memories if x["id"] == mem_ids[0]), None)
            if not m:
                return {
                    "text": f"I don't have a memory saved about that yet. {carer} can add it so we can talk about it next time.",
                    "engine": "template",
                    "sources": [],
                    "why": [],
                    "confidence": "None",
                    "category": None,
                    "alert": None,
                    "people": [],
                    "memory_ids": [],
                    "outcome": "no_memory"
                }, None, ctx

            m_ppl = [people_map[pid] for pid in m.get("people", []) if pid in people_map]
            others = [p for p in m_ppl if p["id"] not in ctx.get("person_ids", [])]
            if others:
                other_strs = [f"{display_name(p['name'])}, your {clean_rel(p['relationship'])}" for p in others]
                verb = "was" if len(others) == 1 else "were"
                text = f"In “{m['title']}”, {', '.join(other_strs)} {verb} there too."
            else:
                names_str = ", ".join(display_name(p["name"]) for p in m_ppl) or "you"
                text = f"That memory only mentions {names_str}."

            new_ctx = {"person_ids": m.get("people", []), "memory_ids": [m["id"]], "fresh": False}
            return {
                "text": text,
                "engine": "template",
                "sources": [{"id": m["id"], "title": m["title"], "people": [p["name"] for p in m_ppl], "date": m.get("memory_date"), "score": 1.0}],
                "why": [{"type": "memory", "memory_id": m["id"], "text": f"“{m['title']}”"}],
                "confidence": "High",
                "category": None,
                "alert": None,
                "people": m.get("people", []),
                "memory_ids": [m["id"]],
                "outcome": "answered"
            }, None, new_ctx

        # 5. WHO IS rule
        if WHO_IS_PATTERN.search(question) and named:
            p = people_map[named[0]]
            desc = f" {p['description'].rstrip('.')}." if p.get("description") else ""
            text = f"{display_name(p['name'])} is your {clean_rel(p['relationship'])}.{desc}"
            new_ctx = {"person_ids": [p["id"]], "memory_ids": [], "fresh": True}
            return {
                "text": text,
                "engine": "template",
                "sources": [],
                "why": [{"type": "person", "person_id": p["id"], "text": f"{display_name(p['name'])} is saved as your {clean_rel(p['relationship'])}"}],
                "confidence": "High",
                "category": None,
                "alert": None,
                "people": [p["id"]],
                "memory_ids": [],
                "outcome": "answered"
            }, None, new_ctx

        # 6. WHERE IS rule (grounded in schedule & reminders)
        if WHERE_IS_PATTERN.search(question) and named:
            p = people_map[named[0]]
            who = display_name(p["name"])
            first_clean = who.replace("Dr. ", "")
            now = datetime.now()
            now_str = now.strftime("%H:%M")
            r_match = next((
                r for r in reminders
                if r.get("active") and r.get("kind") in ("visit", "appointment") and r.get("time_of_day", "") >= now_str
                and re.search(rf"\b{re.escape(first_clean.lower())}\b", r.get("text", "").lower())
            ), None)

            sch = get_person_schedule_now(p["id"], schedules, now)
            parts = []
            why = []
            if sch:
                back_text = f" {who} will be home by {twelve_hour(sch.get('back_at', ''))}." if sch.get("back_at") else ""
                parts.append(f"{who} is {sch['place']} right now.{back_text}")
                why.append({
                    "type": "schedule",
                    "text": f"The caregiver set {who}'s usual schedule: {sch['place']} {sch.get('start_at')}–{sch.get('end_at')}"
                })
            if r_match:
                parts.append(r_match["text"].rstrip(".") + ".")
                why.append({
                    "type": "reminder",
                    "text": f"Today's reminders say: {r_match['text']} ({r_match.get('time_of_day')})"
                })

            if parts:
                out_text = " ".join(parts)
                conf = "High"
            else:
                if who == carer:
                    out_text = f"I don't know where {who} is right now. Let's ask someone nearby to give {who} a call."
                else:
                    out_text = f"I don't know where {who} is right now. {carer} can help you find out."
                why = [{
                    "type": "rule",
                    "text": "EchoBond doesn't track where anyone is, and no visit by this person is saved in today's reminders, so it didn't guess"
                }]
                conf = "None"

            new_ctx = {"person_ids": [p["id"]], "memory_ids": [], "fresh": True}
            return {
                "text": out_text,
                "engine": "template",
                "sources": [],
                "why": why,
                "confidence": conf,
                "category": None,
                "alert": None,
                "people": [p["id"]],
                "memory_ids": [],
                "outcome": "answered"
            }, None, new_ctx

        # 7. Memory Search
        search_q = question + " " + " ".join(people_map[pid]["name"] for pid in person_ids if pid in people_map)
        hits = search_memories(search_q, memories, people_map, places_map, person_ids=person_ids, allow_private=True, exclude_ids=exclude_ids, k=3)
        if person_ids:
            filtered_hits = [h for h in hits if any(pid in h["memory"].get("people", []) for pid in person_ids)]
            if filtered_hits:
                hits = filtered_hits

        if not hits:
            if follow and person_ids:
                names_str = ", ".join(display_name(people_map[pid]["name"]) for pid in person_ids if pid in people_map)
                text = f"That's everything I have saved about {names_str} for now. {carer} can add more."
            else:
                text = f"I don't have a memory saved about that yet. {carer} can add it so we can talk about it next time."
            return {
                "text": text,
                "engine": "template",
                "sources": [],
                "why": [{"type": "rule", "text": "No saved memory matched this question, so EchoBond didn't guess"}],
                "confidence": "None",
                "category": None,
                "alert": None,
                "people": person_ids,
                "memory_ids": [],
                "outcome": "no_memory"
            }, None, ctx

        top = hits[0]
        top_mem = top["memory"]
        leads = ["Here is a lovely memory.", "Let me tell you.", "I remember this one."]
        lead = random.choice(leads)
        content = top_mem.get("content", "").rstrip(".")
        text = f"{lead} {content}."

        score = top["score"]
        conf = "High" if score >= 0.9 else ("Medium" if score >= 0.45 else "Low")

        why_items = []
        if follow:
            why_items.append({"type": "context", "text": "Follow-up question: EchoBond used the people from your last question"})
        for pid in person_ids[:2]:
            if pid in people_map:
                why_items.append({
                    "type": "person",
                    "person_id": pid,
                    "text": f"{display_name(people_map[pid]['name'])} is saved as your {clean_rel(people_map[pid]['relationship'])}"
                })

        when = top_mem.get("memory_date") or top_mem.get("year")
        m_ppl = [people_map[pid] for pid in top_mem.get("people", []) if pid in people_map]
        m_names = [display_name(p["name"]) for p in m_ppl]
        when_part = f" ({when})" if when else ""
        names_joined = ", ".join(m_names)
        mentions_part = f" mentions {names_joined}" if m_names else ""
        why_items.append({
            "type": "memory",
            "memory_id": top_mem["id"],
            "text": "\u201c" + top_mem["title"] + "\u201d" + when_part + mentions_part
        })

        sources = [{
            "id": h["memory"]["id"],
            "title": h["memory"]["title"],
            "people": [people_map[pid]["name"] for pid in h["memory"].get("people", []) if pid in people_map],
            "date": h["memory"].get("memory_date"),
            "score": h["score"]
        } for h in hits]

        new_people_ids = person_ids if person_ids else top_mem.get("people", [])
        new_ctx = {
            "person_ids": new_people_ids,
            "memory_ids": [top_mem["id"]],
            "fresh": not follow
        }

        # 1. Soften any patronizing tone
        text = soften_tone(text)

        # 2. Strict Grounding & Anti-Hallucination Verification
        is_grounded, g_reason, g_meta = verify_grounding(text, people, memories, self.patient_name)
        alert_data = None
        if not is_grounded:
            carer_name = display_name(people[0]["name"]) if people else "your family"
            text = f"I'm keeping our conversation safe and simple. {carer_name} is close by if you'd like to ask them."
            alert_data = {
                "level": "warning",
                "kind": "grounding_violation",
                "reason": "Safety filter blocked unverified fact/text",
                "detail": f"Blocked response: {g_reason}"
            }
            why_items.append({"type": "rule", "text": "Safety filter intercepted response to prevent ungrounded fact/advice"})
        else:
            why_items.append({"type": "grounding", "text": "Verified: 100% grounded in caregiver-saved memories (zero hallucination)"})

        return {
            "text": text,
            "engine": "template",
            "sources": sources,
            "why": why_items,
            "confidence": conf,
            "category": None,
            "alert": alert_data.get("level") if alert_data else None,
            "grounding": {
                "verified": is_grounded,
                "zero_hallucination": is_grounded,
                "notes": g_reason
            },
            "people": new_people_ids,
            "memory_ids": [top_mem["id"]],
            "outcome": "answered" if is_grounded else "safety"
        }, alert_data, new_ctx
