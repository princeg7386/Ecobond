"""
EchoBond Persona & Conversational Intelligence Service
======================================================
Provides natural, warm, first-person family voice responses and intelligent
dialogue generation for Alzheimer's patients.

Architecture:
1. Gemini Generative AI (if GEMINI_API_KEY is configured in env or SQLite settings):
   - Fully grounded in real family memories, schedule, location, and relationship.
   - Strictly constrained: warm, patient, concise (1-3 sentences), reassuring.
2. Rich Semantic Persona Engine (Built-in, zero API key required):
   - First-person familial dialogue ("Nani", "Ma", "Dadi", "I", "Main").
   - Intent recognition across Hindi, Hinglish, and English.
   - Dynamic schedule, occupation, time-of-day, and memory awareness.
"""

import os
import re
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


def get_gemini_api_key(db_conn=None) -> Optional[str]:
    """Retrieve Gemini API key from environment or database settings."""
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    if db_conn:
        try:
            row = db_conn.execute("SELECT value FROM settings WHERE key = 'gemini_api_key'").fetchone()
            if row and row[0] and row[0].strip():
                return row[0].strip()
        except Exception:
            pass
    return None


def call_gemini(api_key: str, prompt: str, system_instruction: str = "") -> Optional[str]:
    """Call Google Gemini REST API using standard urllib."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200,
            "topP": 0.9,
        }
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception as e:
        print(f"[PersonaService] Gemini API call error: {e}")
    return None


def get_patient_salutation(person: Dict[str, Any], patient_name: str = "Kamala") -> str:
    """Get what the person calls the patient (e.g. Nani, Ma, Dadi)."""
    calls = (person.get("calls_patient") or "").strip()
    if calls:
        return calls
    rel = (person.get("relationship") or "").lower()
    if "grand" in rel:
        return "Nani"
    if "daughter" in rel or "son" in rel or "child" in rel:
        return "Ma"
    if "sister" in rel:
        return "Didi"
    return patient_name


def generate_family_voice_reply(
    message: str,
    person: Dict[str, Any],
    patient_name: str,
    schedule: Optional[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    db_conn=None
) -> Tuple[str, str]:
    """
    Generate a warm, natural, first-person response from the family member.
    Returns: (reply_text, engine_used: 'gemini' | 'persona_engine')
    """
    first_name = person["name"].split()[0]
    salutation = get_patient_salutation(person, patient_name)
    now = datetime.now()
    time_str = now.strftime("%I:%M %p")
    day_str = now.strftime("%A")

    # 1. Try Gemini LLM if key is present
    api_key = get_gemini_api_key(db_conn)
    if api_key:
        sched_info = (
            f"You are currently at {schedule['place']}. You plan to be home by {schedule.get('back_at', 'this evening')}."
            if schedule else f"You work as a {person.get('occupation') or 'professional'}. It is currently {time_str} on {day_str}."
        )
        person_mems = [m for m in memories if person["id"] in m.get("people", [])][:3]
        mem_summary = " ; ".join(f"{m['title']}: {m.get('content', '')[:100]}" for m in person_mems)

        system_prompt = (
            f"You are roleplaying as {person['name']}, who is the {person['relationship']} to {patient_name} (an elderly person with Alzheimer's/memory loss). "
            f"You address her affectionately as '{salutation}'. "
            f"Schedule/status: {sched_info} "
            f"Shared family memories: {mem_summary or 'No specific memory saved yet.'} "
            f"\nGuidelines:\n"
            f"1. Respond directly to {salutation} in the FIRST PERSON ('I', 'Main', 'Mujhe'). Never talk about yourself in the 3rd person.\n"
            f"2. Tone: deeply loving, reassuring, calm, respectful Indian family style.\n"
            f"3. Language: Match her question — use warm Hinglish/Hindi or gentle English.\n"
            f"4. Length: Keep it SHORT (1 to 3 sentences maximum), suitable to be synthesized into voice and easily understood by someone with memory loss.\n"
            f"5. Medical/Safety: Never contradict reality, never cause alarm, never promise unrealistic things."
        )

        user_prompt = f"{salutation} asks: \"{message}\"\nReply as {first_name}:"
        llm_reply = call_gemini(api_key, user_prompt, system_prompt)
        if llm_reply:
            # Clean up quotation marks
            llm_reply = re.sub(r'^["\']|["\']$', '', llm_reply.strip())
            return llm_reply, "gemini"

    # 2. Rich Semantic Persona Engine (Fallback / Zero-config)
    low = message.lower().strip()
    occ = (person.get("occupation") or "").lower()

    # Detect language preference (Hindi/Hinglish vs pure English)
    is_hindi = bool(re.search(r"\b(kahan|kaise|kaisa|kya|kyun|kab|ho|hai|hu|aap|mera|meri|mujhe|yaad|khana|dawa|ghar)\b", low))

    # A. Where are you / Coming home
    if re.search(r"\b(where|kahan|kab\s+aa|coming\s+home|when\s+will\s+you|reach|office|ghar\s+kab)\b", low):
        if schedule:
            place = schedule.get("place", "office")
            back_at = schedule.get("back_at", "")
            if is_hindi:
                back_phrase = f"Lagbhag {back_at} tak ghar laut aaunga." if back_at else "Shaam tak ghar aaunga."
                return f"Namaste {salutation}! Main abhi {place} pe hoon. {back_phrase} Aap bilkul chinta mat karna, main jaldi milunga!", "persona_engine"
            else:
                back_phrase = f"I'll be home around {back_at}." if back_at else "I'll be home this evening."
                return f"Hello {salutation}! I am at {place} right now. {back_phrase} Please rest and don't worry, I'll see you soon!", "persona_engine"
        else:
            # Contextual based on occupation & time of day
            hour = now.hour
            if "librarian" in occ or "school" in occ:
                place = "school library" if 9 <= hour <= 16 else "nearby"
            elif "student" in occ or "art" in occ:
                place = "college art studio" if 9 <= hour <= 17 else "study session"
            else:
                place = "office" if 9 <= hour <= 18 else "market"

            if is_hindi:
                return f"Main abhi {place} pe hoon {salutation}, thoda kaam nipat raha hai. Shaam ko jaldi ghar aaunga. Aapne chai pi na?", "persona_engine"
            else:
                return f"I'm at the {place} right now, {salutation}, just wrapping up my work. I'll be home soon. Make sure you take some rest!", "persona_engine"

    # B. Missing you / Affection / Love
    if re.search(r"\b(miss|missing|yaad|love|pyar|alone|lonely|mann\s+nahi|udas)\b", low):
        if is_hindi:
            return f"Aww {salutation}, mujhe bhi aapki bohot yaad aa rahi hai! Main hamesha aapke saath hoon. Main jaldi aapse milne aaunga aur hum dher saari baatein karenge.", "persona_engine"
        else:
            return f"I miss you so much too, {salutation}! You are always in my thoughts. I'll be with you very soon, so please keep smiling for me!", "persona_engine"

    # C. How was your day / What are you doing
    if re.search(r"\b(how\s+was\s+your\s+day|what\s+are\s+you\s+doing|din\s+kaisa|kya\s+kar\s+rahe|kaam\s+kaisa)\b", low):
        if "student" in occ or "art" in occ:
            detail_hi = "aaj class mein ek nayi painting banayi, agli baar aapko dikhaungi!"
            detail_en = "working on a new sketch in class today, I can't wait to show it to you!"
        elif "librarian" in occ:
            detail_hi = "library mein nayi kitabein organise kar rahi thi."
            detail_en = "organising new books at the library today."
        else:
            detail_hi = "office mein software project pe kaam chal raha hai."
            detail_en = "working on a software project at the office today."

        if is_hindi:
            return f"Mera din kaafi achha raha {salutation}! Main {detail_hi} Aapka din kaisa bita? Aapne aaram kiya na?", "persona_engine"
        else:
            return f"My day has been really good, {salutation}! I was {detail_en} How was your day? Did you get some rest?", "persona_engine"

    # D. Food / Medication / Health
    if re.search(r"\b(eat|lunch|dinner|khana|food|dawa|medicine|tabiyat|health|bhookh)\b", low):
        if is_hindi:
            return f"Haan {salutation}, maine time par khana kha liya tha! Aur aapne lunch aur doctor ki dawa li na? Apna khayal achhe se rakhna.", "persona_engine"
        else:
            return f"Yes {salutation}, I had my meal on time! Did you have your lunch and take your medicine today? Please take good care of yourself.", "persona_engine"

    # E. Memories / Past Stories
    if re.search(r"\b(memory|remember|yaad\s+hai|purani|what\s+did\s+we\s+do|together|kaha\s+gaye)\b", low):
        person_mems = [m for m in memories if person["id"] in m.get("people", [])]
        if person_mems:
            mem = person_mems[0]
            title = mem.get("title", "")
            content = mem.get("content", "").rstrip(".")
            if is_hindi:
                return f"Haan {salutation}, mujhe '{title}' bohot achhi tarah yaad hai! {content}. Kitna pyara samay tha na hum sabka saath mein!", "persona_engine"
            else:
                return f"Yes {salutation}, I remember '{title}' so clearly! {content}. That was such a wonderful day together.", "persona_engine"
        else:
            if is_hindi:
                return f"Haan {salutation}, mujhe aapke saath bitaye saare pal bohot yaad aate hain. Hum jaldi baithkar purani baatein karenge!", "persona_engine"
            else:
                return f"Yes {salutation}, I cherish all our happy times together. When I see you next, we'll look through all the family photos together!", "persona_engine"

    # F. Greetings / Kaise ho / How are you
    if re.search(r"\b(hello|hi|namaste|kaise\s+ho|how\s+are\s+you|kya\s+haal|pranam)\b", low):
        if is_hindi:
            return f"Namaste {salutation}! Main bilkul theek aur khush hoon. Aap kaisi ho? Aapki aawaz sunkar bohot achha laga.", "persona_engine"
        else:
            return f"Hello {salutation}! I'm doing really well, thank you. How are you feeling today? It makes me so happy to hear from you.", "persona_engine"

    # G. Distress / Scared / Home
    if re.search(r"\b(scared|darr|lost|kahan\s+hu|who\s+am\s+i|panic|help)\b", low):
        if is_hindi:
            return f"Ghabrao mat {salutation}, aap apne pyare ghar mein ho, bilkul surakshit ho. Hum sab aapke saath hain. Ek lambi saans lo, sab theek hai.", "persona_engine"
        else:
            return f"Please don't worry, {salutation}. You are in your safe, loving home. We all love you and are right here for you. Everything is okay.", "persona_engine"

    # H. Default conversational reply in first-person
    if is_hindi:
        return f"Main aapki baat samajh raha hoon {salutation}. Main hamesha aapke saath hoon. Kuch aur puchna hai to batayein, main sun raha hoon.", "persona_engine"
    else:
        return f"I hear you, {salutation}. I am always here with you. What else is on your mind today? Tell me, I'm listening.", "persona_engine"
