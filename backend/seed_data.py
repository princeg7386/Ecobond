import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import random

from .config import DB_PATH, ASSETS_DIR, PATIENT_NAME
from .database import get_db, init_db, now_iso
from .services.face_service import extract_embedding

def seed_database(force: bool = False):
    init_db()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM people")
    count = cursor.fetchone()[0]
    if count > 0 and not force:
        conn.close()
        return

    # Clear existing data if force
    if force:
        tables = [
            "faces", "memory_people", "memories", "places", "relations",
            "reminders", "reminder_log", "alerts", "interactions", "notes",
            "audit", "schedules", "voice_messages", "voices", "people"
        ]
        for t in tables:
            cursor.execute(f"DELETE FROM {t}")
        conn.commit()

    # Load demo.json from embedded assets or file
    demo_json_path = ASSETS_DIR / "family" / "demo.json"
    if demo_json_path.exists():
        with open(demo_json_path, "r", encoding="utf-8") as f:
            demo_data = json.load(f)
    else:
        # Extract from index.html if demo.json not standalone
        index_path = ASSETS_DIR.parent / "index.html"
        import re
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
            m = re.search(r'window\.EB_DEMO_DATA\s*=\s*(\{.*?\});', content)
            if m:
                demo_data = json.loads(m.group(1))
            else:
                demo_data = {}

    patient = demo_data.get("patient", PATIENT_NAME)
    cursor.execute("UPDATE settings SET value = ? WHERE key = 'patient_name'", (patient,))

    key_to_person_id = {}
    profiles = demo_data.get("profiles", {})
    schedules = demo_data.get("schedule", {})
    voice_messages = demo_data.get("voice_messages", {})
    voice_consent_keys = demo_data.get("voice_consent", []) or ["rohan", "priya", "anjali"]

    default_pitches = {
        "rohan": 135.0,
        "priya": 225.0,
        "anjali": 195.0,
        "vikram": 115.0,
        "sarla": 205.0,
        "lakshmi": 210.0,
        "mehta": 215.0,
        "ramesh": 120.0,
    }

    # Insert People
    for p in demo_data.get("people", []):
        key = p.get("key")
        prof = profiles.get(key, {})
        has_photo = bool(p.get("photo"))
        consent_photo = 1 if has_photo else 0
        consent_voice = 1 if key in voice_consent_keys else 0
        photo_url = f"/assets/family/{key}_0.png" if has_photo else None
        pitch = prof.get("voice_pitch") or default_pitches.get(key)

        cursor.execute("""
        INSERT INTO people (
            key, name, relationship, nickname, description, status,
            consent_photo, consent_voice_clone, consent_given_by, consent_date,
            avatar_color, photo, has_voice, birthday, favourites, notes,
            occupation, hobbies, connection, topics, calls_patient, voice_pitch, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            p.get("name"),
            p.get("relationship"),
            p.get("nickname", ""),
            p.get("description", ""),
            p.get("status", "active"),
            consent_photo,
            consent_voice,
            "Fictional demo data" if has_photo else "",
            now_iso() if has_photo else "",
            p.get("color", "#8fa3b8"),
            photo_url,
            consent_voice,
            prof.get("birthday", ""),
            prof.get("favourites", ""),
            prof.get("notes", ""),
            prof.get("occupation", ""),
            prof.get("hobbies", ""),
            prof.get("connection", ""),
            prof.get("topics", ""),
            prof.get("calls_patient", ""),
            pitch,
            now_iso()
        ))
        person_id = cursor.lastrowid
        key_to_person_id[key] = person_id

        # Insert Enrolled Faces
        if has_photo:
            for variant in (0, 1):
                img_path = ASSETS_DIR / "family" / f"{key}_{variant}.png"
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as img_f:
                            emb = extract_embedding(img_f.read())
                        cursor.execute("""
                        INSERT INTO faces (person_id, embedding, photo_url, created_at)
                        VALUES (?, ?, ?, ?)
                        """, (person_id, json.dumps(emb), f"/assets/family/{key}_{variant}.png", now_iso()))
                    except Exception as e:
                        print(f"Error enrolling face {img_path}: {e}")

        # Insert Schedule
        for sch in schedules.get(key, []):
            cursor.execute("""
            INSERT INTO schedules (person_id, days, start_at, end_at, place, back_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                person_id,
                sch.get("days", "daily"),
                sch.get("start_at", sch.get("start", "")),
                sch.get("end_at", sch.get("end", "")),
                sch.get("place", sch.get("where", "")),
                sch.get("back_at", "")
            ))

        # Insert Voice Messages
        for tpl in voice_messages.get(key, []):
            cursor.execute("""
            INSERT INTO voice_messages (person_id, kind, template, approved_by, approved_at, active)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                person_id,
                "away",
                tpl,
                f"{p.get('name')} (themself, fictional demo)",
                now_iso(),
                1
            ))

    # Insert Places
    key_to_place_id = {}
    for pl in demo_data.get("places", []):
        cursor.execute("""
        INSERT INTO places (name, region, lat, lng, note)
        VALUES (?, ?, ?, ?, ?)
        """, (
            pl.get("name"),
            pl.get("region", ""),
            pl.get("lat"),
            pl.get("lng"),
            pl.get("note", "")
        ))
        place_id = cursor.lastrowid
        key_to_place_id[pl.get("key")] = place_id

    # Insert Memories
    for m in demo_data.get("memories", []):
        place_id = key_to_place_id.get(m.get("place"))
        song = m.get("song") or {}
        cursor.execute("""
        INSERT INTO memories (
            title, content, memory_date, year, life_stage, category,
            place_id, privacy, song_title, song_artist, song_url, photo_url, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            m.get("title"),
            m.get("content"),
            m.get("date", ""),
            m.get("year"),
            m.get("life_stage", "recent"),
            m.get("category", "family"),
            place_id,
            m.get("privacy", "normal"),
            song.get("title", ""),
            song.get("artist", ""),
            None,
            None,
            "demo",
            now_iso()
        ))
        mem_id = cursor.lastrowid

        for k in m.get("people", []):
            if k in key_to_person_id:
                cursor.execute("""
                INSERT OR IGNORE INTO memory_people (memory_id, person_id)
                VALUES (?, ?)
                """, (mem_id, key_to_person_id[k]))

    # Insert Relations
    for r in demo_data.get("relations", []):
        p1 = key_to_person_id.get(r.get("person"))
        p2 = key_to_person_id.get(r.get("related"))
        if p1 and p2:
            cursor.execute("""
            INSERT INTO relations (person_id, related_id, kind)
            VALUES (?, ?, ?)
            """, (p1, p2, r.get("kind")))

    # Insert Reminders
    for rem in demo_data.get("reminders", []):
        cursor.execute("""
        INSERT INTO reminders (text, time_of_day, kind, days, active)
        VALUES (?, ?, ?, ?, ?)
        """, (
            rem.get("text"),
            rem.get("time"),
            rem.get("kind", "routine"),
            "daily",
            1
        ))

    # Seed Activity History & Interactions
    now_dt = datetime.utcnow()
    def at_dt(days_ago: float, hours_ago: float) -> str:
        t = now_dt - timedelta(days=days_ago, hours=hours_ago)
        return t.isoformat() + "Z"

    living_keys = ["anjali", "rohan", "priya", "lakshmi", "mehta"]
    living_ids = [key_to_person_id[k] for k in living_keys if k in key_to_person_id]
    
    qs = [
        ("Who is Rohan?", "answered"),
        ("Tell me about the mango pickle", "answered"),
        ("When is Dr. Mehta coming?", "answered"),
        ("Where did I keep my glasses?", "no_memory"),
        ("What was my school called?", "no_memory")
    ]

    for d in range(6, 0, -1):
        for _ in range(5):
            h = random.uniform(1, 12)
            when = at_dt(d, h)
            x = random.random()
            if x < 0.72 and living_ids:
                pid = random.choice(living_ids)
                score = round(0.9 + random.random() * 0.07, 4)
                cursor.execute("""
                INSERT INTO interactions (kind, person_id, outcome, score, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, ("recognize", pid, "recognized", score, "", when))
            elif x < 0.86:
                pid = key_to_person_id.get("mehta" if random.random() < 0.6 else "priya")
                score = round(0.76 + random.random() * 0.1, 4)
                cursor.execute("""
                INSERT INTO interactions (kind, person_id, outcome, score, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, ("recognize", pid, "uncertain", score, "No name given", when))
            else:
                score = round(0.5 + random.random() * 0.2, 4)
                cursor.execute("""
                INSERT INTO interactions (kind, person_id, outcome, score, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, ("recognize", None, "unknown", score, "Unknown face", when))

        q_item, o = random.choice(qs)
        norm = " ".join(sorted(re.findall(r"[a-z0-9]+", q_item.lower())))
        cursor.execute("""
        INSERT INTO interactions (kind, outcome, question, normalized, summary, memory_ids, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("chat", o, q_item, norm, f"Q: {q_item}", "[]", at_dt(d, 3)))

    # Today's interactions
    for k, h in [("anjali", 0.6), ("priya", 1.4), ("rohan", 2.1)]:
        if k in key_to_person_id:
            cursor.execute("""
            INSERT INTO interactions (kind, person_id, outcome, score, summary, memory_ids, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("recognize", key_to_person_id[k], "recognized", 0.95, "", "[]", at_dt(0, h)))

    if "mehta" in key_to_person_id:
        cursor.execute("""
        INSERT INTO interactions (kind, person_id, outcome, score, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, ("recognize", key_to_person_id["mehta"], "uncertain", 0.84, "No name given", at_dt(0, 1.1)))

    for h in (5.5, 4.0, 2.2):
        cursor.execute("""
        INSERT INTO interactions (kind, outcome, question, normalized, summary, memory_ids, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("chat", "answered", "Where is Priya?", "priya where", "Q: Where is Priya?", "[]", at_dt(0, h)))

    # Safety questions
    safety_seeds = [
        (5, 4, "I feel so lonely today", "sadness"),
        (3, 2, "I want to go home", "confusion"),
        (3, 1, "Leave me alone", "frustration"),
        (0, 2, "I'm scared, nobody is here", "fear")
    ]
    for d, h, q_txt, cat in safety_seeds:
        norm = " ".join(sorted(re.findall(r"[a-z0-9]+", q_txt.lower())))
        cursor.execute("""
        INSERT INTO interactions (kind, outcome, question, normalized, category, summary, memory_ids, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("chat", "safety", q_txt, norm, cat, f"Q: {q_txt}", "[]", at_dt(d, h)))

    # Notes
    notes_seed = [
        (6, "good", "Sang with Priya, ate well."),
        (5, "okay", "Tired after lunch."),
        (4, "good", "Long walk with Lakshmi."),
        (3, "hard", "Restless in the evening, asked to go home."),
        (2, "okay", ""),
        (1, "good", "Recognised Rohan on the video call straight away.")
    ]
    for d, mood, note_text in notes_seed:
        cursor.execute("""
        INSERT INTO notes (mood, note, author, created_at)
        VALUES (?, ?, ?, ?)
        """, (mood, note_text, "Anjali", at_dt(d, 3)))

    # Alerts
    cursor.execute("""
    INSERT INTO alerts (level, kind, reason, detail, acknowledged, ref, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("warning", "distress", "Possible distress in conversation", "I want to go home", 1, "confusion", at_dt(3, 2)))

    cursor.execute("""
    INSERT INTO alerts (level, kind, reason, detail, acknowledged, ref, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("warning", "protected_person", "Asked about a protected person", "Where is Ramesh?", 0, "", at_dt(1, 2)))

    cursor.execute("""
    INSERT INTO alerts (level, kind, reason, detail, acknowledged, ref, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("warning", "distress", "Possible distress in conversation", "I'm scared, nobody is here", 0, "fear", at_dt(0, 0.8)))

    cursor.execute("""
    INSERT INTO alerts (level, kind, reason, detail, acknowledged, ref, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("info", "repeated_question", "Repeated question", 'Asked 3 times today: "Where is Priya?"', 0, "priya where", at_dt(0, 2.1)))

    # Audit log
    audit_seed = [
        ("seed_demo", "", 14),
        ("create_person", "person:1", 14),
        ("add_voice", "person:1", 12),
        ("update_person", "person:6:status", 9),
        ("create_memory", "memory:9", 3)
    ]
    for action, target, d in audit_seed:
        cursor.execute("""
        INSERT INTO audit (actor, action, target, created_at)
        VALUES (?, ?, ?, ?)
        """, ("system" if action == "seed_demo" else "caregiver", action, target, at_dt(d, 0)))

    conn.commit()
    conn.close()
    print("Database seeded successfully with family profiles, faces, memories, reminders, and history.")

if __name__ == "__main__":
    seed_database(force=True)
