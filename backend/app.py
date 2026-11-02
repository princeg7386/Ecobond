"""
EchoBond Backend – Flask API Server
====================================
A real Python backend replacing the in-browser mock.
All API routes match the frontend's EB.api interface exactly.

Run:  python -m backend.app
"""
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, abort, g, Response

from .config import BASE_DIR, BACKEND_DIR, DATA_DIR, UPLOADS_DIR, ASSETS_DIR, DB_PATH, DEFAULT_PIN, PATIENT_NAME, PORT, HOST
from .database import get_db, init_db, now_iso
from .seed_data import seed_database
from .services.face_service import extract_embedding, match_face, cosine_similarity, check_image_quality
from .services.safety import detect_support_category, build_support_reply, ALERT_LEVELS
from .services.memory_store import search_memories, tokenize, normalize_question
from .services.conversation import (
    ConversationEngine, display_name, clean_rel, twelve_hour,
    day_matches, get_person_schedule_now
)
from .services.companion import browser_voice_params, make_segment, TALK_PATTERN, WHERE_PATTERN
from .services.voice_service import (
    analyze_audio_bytes, convert_to_wav, build_voice_profile,
    speak_as_person, speak_as_echobond, synthesize_speech_sync,
    SYSTEM_VOICE, SYSTEM_PITCH, SYSTEM_RATE,
)
from .services.persona_service import (
    generate_family_voice_reply, get_gemini_api_key, call_gemini
)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

PRIVACY_LEVELS = ["normal", "private", "sensitive", "locked"]

# In-memory session context for conversations
sessions = {}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def db():
    """Get a database connection for the current request."""
    if "db" not in g:
        g.db = get_db()
    return g.db


@app.teardown_appcontext
def close_db(exception):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def setting(key):
    row = db().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    defaults = {
        "repeat_window_hours": "24",
        "repeat_threshold": "3",
        "voice_lang": "en-IN",
        "voice_rate": "0.86",
        "default_privacy": "normal",
        "face_recognize_threshold": "0.90",
        "pin": DEFAULT_PIN,
        "patient_name": PATIENT_NAME,
    }
    return (row["value"] if row else defaults.get(key, ""))


def patient_name():
    return setting("patient_name")


def audit(action, target=""):
    db().execute(
        "INSERT INTO audit (actor, action, target, created_at) VALUES (?, ?, ?, ?)",
        ("caregiver", action, target, now_iso()),
    )
    db().commit()


def add_alert(level, kind, reason, detail, ref=""):
    db().execute(
        "INSERT INTO alerts (level, kind, reason, detail, acknowledged, ref, created_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
        (level, kind, reason, detail, ref, now_iso()),
    )
    db().commit()


def caregiver_required(f):
    """Decorator: check X-Caregiver-Pin header."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        pin = request.headers.get("X-Caregiver-Pin", "")
        if pin != setting("pin"):
            return jsonify({"error": "Caregiver PIN is missing or incorrect."}), 401
        return f(*args, **kwargs)
    return wrapper


def is_caregiver():
    return request.headers.get("X-Caregiver-Pin", "") == setting("pin")


def person_row(person_id):
    return db().execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()


def person_public(row, full=False):
    p = dict(row)
    o = {
        "id": p["id"], "name": p["name"], "relationship": p["relationship"],
        "nickname": p["nickname"], "avatar_color": p["avatar_color"], "status": p["status"],
        "photo_url": p["photo"] if p["consent_photo"] and p["status"] == "active" else None,
    }
    if full:
        face_count = db().execute("SELECT COUNT(*) as c FROM faces WHERE person_id = ?", (p["id"],)).fetchone()["c"]
        mem_count = db().execute("SELECT COUNT(*) as c FROM memory_people WHERE person_id = ?", (p["id"],)).fetchone()["c"]
        o.update({
            "description": p["description"],
            "consent_photo": p["consent_photo"],
            "consent_voice_clone": p["consent_voice_clone"],
            "consent_given_by": p["consent_given_by"],
            "consent_date": p["consent_date"],
            "birthday": p["birthday"],
            "favourites": p["favourites"],
            "notes": p["notes"],
            "occupation": p["occupation"],
            "hobbies": p["hobbies"],
            "connection": p["connection"],
            "topics": p["topics"],
            "calls_patient": p["calls_patient"],
            "face_samples": face_count,
            "has_voice": bool(p["has_voice"]),
            "memory_count": mem_count,
            "photo_url": p["photo"] if p["consent_photo"] else None,
        })
    return o


def memory_people_ids(mem_id):
    rows = db().execute("SELECT person_id FROM memory_people WHERE memory_id = ?", (mem_id,)).fetchall()
    return [r["person_id"] for r in rows]


def mem_public(row, carer=False, unlocked=False):
    m = dict(row)
    hidden = m["privacy"] == "locked" and not unlocked
    ppl_ids = memory_people_ids(m["id"])
    ppl = []
    for pid in ppl_ids:
        pr = person_row(pid)
        if pr:
            ppl.append({"id": pr["id"], "name": pr["name"], "relationship": pr["relationship"]})

    place = None
    if m["place_id"]:
        pl = db().execute("SELECT * FROM places WHERE id = ?", (m["place_id"],)).fetchone()
        if pl:
            place = {"id": pl["id"], "name": pl["name"], "region": pl["region"]}

    song = None
    if m["song_title"]:
        song = {"title": m["song_title"], "artist": m["song_artist"], "audio_url": m["song_url"]}

    o = {
        "id": m["id"], "title": m["title"],
        "content": None if hidden else m["content"],
        "locked": hidden,
        "memory_date": m["memory_date"], "year": m["year"],
        "life_stage": m["life_stage"], "category": m["category"],
        "privacy": m["privacy"], "people": ppl, "place": place,
        "song": song, "photo_url": m["photo_url"], "created_at": m["created_at"],
        "person_id": ppl[0]["id"] if ppl else None,
        "person_name": ppl[0]["name"] if ppl else None,
        "sensitive": 1 if m["privacy"] in ("sensitive", "locked") else 0,
    }
    if carer:
        o["source"] = m["source"]
    return o


def load_all_people():
    return [dict(r) for r in db().execute("SELECT * FROM people").fetchall()]


def load_all_memories():
    mems = []
    for r in db().execute("SELECT * FROM memories").fetchall():
        m = dict(r)
        m["people"] = memory_people_ids(m["id"])
        mems.append(m)
    return mems


def load_all_places():
    return [dict(r) for r in db().execute("SELECT * FROM places").fetchall()]


def load_all_reminders():
    return [dict(r) for r in db().execute("SELECT * FROM reminders").fetchall()]


def load_all_schedules():
    return [dict(r) for r in db().execute("SELECT * FROM schedules").fetchall()]


def get_enrolled_faces():
    """Returns {person_id: [[embedding], ...]}"""
    faces = {}
    for r in db().execute("SELECT person_id, embedding FROM faces").fetchall():
        pid = r["person_id"]
        emb = json.loads(r["embedding"])
        faces.setdefault(pid, []).append(emb)
    return faces


def get_carer_display():
    people = load_all_people()
    carer = next((p for p in people if "caregiver" in p.get("relationship", "").lower() and p.get("status") == "active"), None)
    if not carer:
        carer = next((p for p in people if p.get("status") == "active"), None)
    return display_name(carer["name"]) if carer else "your family"


def same_local_day(iso_str):
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date() == datetime.utcnow().date()
    except Exception:
        return False


def generation(rel_str):
    r = rel_str.lower()
    if re.search(r"grandson|granddaughter|grandchild", r): return 2
    if re.search(r"son|daughter|child|nephew|niece", r): return 1
    if re.search(r"mother|father|parent", r): return -1
    if re.search(r"husband|wife|spouse|sister|brother|sibling|cousin", r): return 0
    return None


# ──────────────────────────────────────────────
# Static file serving (the frontend)
# ──────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/assets/<path:path>")
def serve_assets(path):
    return send_from_directory(str(ASSETS_DIR), path)


@app.route("/uploads/<path:path>")
def serve_uploads(path):
    return send_from_directory(str(UPLOADS_DIR), path)


# ──────────────────────────────────────────────
# /api/health
# ──────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "app": "EchoBond",
        "demo": False,
        "version": "2.0.0",
        "patient": patient_name(),
        "components": {
            "face": "PIL + numpy (server)",
            "face_production_quality": False,
            "caption": "unavailable",
            "text_embeddings": "keyword (server)",
            "vector_store": "SQLite",
            "llm": "template (offline, no LLM)",
            "speech_to_text": "browser",
            "speaker_id": "unavailable",
            "tts": "browser",
        },
        "encryption_at_rest": False,
    })


# ──────────────────────────────────────────────
# /api/people
# ──────────────────────────────────────────────
@app.route("/api/people")
def api_people():
    carer = is_caregiver()
    rows = db().execute("SELECT * FROM people").fetchall()
    if not carer:
        rows = [r for r in rows if r["status"] == "active"]
    return jsonify([person_public(r, carer) for r in rows])


@app.route("/api/people/<int:pid>")
@caregiver_required
def api_person_detail(pid):
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404
    o = person_public(p, True)
    rels = db().execute(
        "SELECT r.*, p1.name as person_name, p2.name as related_name FROM relations r "
        "LEFT JOIN people p1 ON r.person_id=p1.id LEFT JOIN people p2 ON r.related_id=p2.id "
        "WHERE r.person_id=? OR r.related_id=?", (pid, pid)
    ).fetchall()
    o["relations"] = [dict(r) for r in rels]
    recent = db().execute(
        "SELECT kind, outcome, created_at FROM interactions WHERE person_id=? ORDER BY id DESC LIMIT 5", (pid,)
    ).fetchall()
    o["recent"] = [dict(r) for r in recent]
    return jsonify(o)


@app.route("/api/people", methods=["POST"])
@caregiver_required
def api_create_person():
    f = request.form
    name = (f.get("name") or "").strip()
    rel_txt = (f.get("relationship") or "").strip()
    if not name or not rel_txt:
        return jsonify({"error": "Name and relationship are required."}), 400

    cp = f.get("consent_photo") == "1"
    cv = f.get("consent_voice_clone") == "1"
    by = (f.get("consent_given_by") or "").strip()
    if (cp or cv) and not by:
        return jsonify({"error": "Record who gave consent (the person themself, or their guardian)."}), 400

    conn = db()
    cur = conn.execute(
        """INSERT INTO people (key, name, relationship, nickname, description, status,
        consent_photo, consent_voice_clone, consent_given_by, consent_date,
        avatar_color, photo, has_voice, birthday, favourites, notes,
        occupation, hobbies, connection, topics, calls_patient, voice_pitch, created_at)
        VALUES ('', ?, ?, ?, ?, 'active', ?, ?, ?, ?, '#c9a36b', NULL, 0, ?, ?, ?, '', '', '', '', '', NULL, ?)""",
        (name, rel_txt, f.get("nickname", ""), f.get("description", ""),
         1 if cp else 0, 1 if cv else 0, by, now_iso() if by else "",
         f.get("birthday", ""), f.get("favourites", ""), f.get("notes", ""), now_iso())
    )
    pid = cur.lastrowid
    conn.commit()

    photos_result = []
    for photo_file in request.files.getlist("photos")[:3]:
        if not cp:
            photos_result.append({"enrolled": False, "reason": "No photo consent recorded for this person."})
            continue
        img_bytes = photo_file.read()
        try:
            emb = extract_embedding(img_bytes)
            fname = f"person_{pid}_{len(photos_result)}.png"
            fpath = UPLOADS_DIR / fname
            with open(fpath, "wb") as wf:
                wf.write(img_bytes)
            photo_url = f"/uploads/{fname}"
            conn.execute(
                "INSERT INTO faces (person_id, embedding, photo_url, created_at) VALUES (?, ?, ?, ?)",
                (pid, json.dumps(emb), photo_url, now_iso())
            )
            # Set first photo as avatar
            if not photos_result or all(not pr.get("enrolled") for pr in photos_result):
                conn.execute("UPDATE people SET photo = ? WHERE id = ?", (photo_url, pid))
            photos_result.append({"enrolled": True})
        except Exception as e:
            photos_result.append({"enrolled": False, "reason": str(e)})
    conn.commit()

    voice_result = None
    voice_file = request.files.get("voice")
    if voice_file and cv:
        try:
            v_bytes = voice_file.read()
            if len(v_bytes) >= 1000:
                wav_b = convert_to_wav(v_bytes, voice_file.filename or "")
                analysis = analyze_audio_bytes(wav_b)
                if analysis.get("ok"):
                    fname = f"voice_person_{pid}.wav"
                    fpath = UPLOADS_DIR / fname
                    with open(fpath, "wb") as wf:
                        wf.write(v_bytes)
                    v_url = f"/uploads/{fname}"
                    conn.execute("INSERT INTO voices (person_id, audio_url, created_at) VALUES (?, ?, ?)", (pid, v_url, now_iso()))
                    conn.execute("UPDATE people SET has_voice = 1, voice_pitch = ? WHERE id = ?", (analysis.get("pitch_hz"), pid))
                    conn.commit()
                    voice_result = {"stored": True, "analysis": analysis, "audio_url": v_url}
        except Exception as ve:
            voice_result = {"stored": False, "error": str(ve)}

    audit("create_person", f"person:{pid}")
    p = person_row(pid)
    return jsonify({"person": person_public(p, True), "photos": photos_result, "voice": voice_result}), 201


@app.route("/api/people/<int:pid>", methods=["PATCH"])
@caregiver_required
def api_update_person(pid):
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404
    b = request.get_json(force=True, silent=True) or {}
    allowed = ["name", "relationship", "nickname", "description", "status",
               "consent_photo", "consent_voice_clone", "consent_given_by",
               "birthday", "favourites", "notes", "occupation", "hobbies",
               "connection", "topics", "calls_patient"]
    keys = [k for k in b if k in allowed]
    if not keys:
        return jsonify({"error": "Nothing to update."}), 400
    if b.get("status") and b["status"] not in ("active", "deceased", "do_not_surface"):
        return jsonify({"error": "Status must be active, deceased or do_not_surface."}), 400

    conn = db()
    for k in keys:
        conn.execute(f"UPDATE people SET {k} = ? WHERE id = ?", (b[k], pid))

    if "consent_photo" in b and not int(b["consent_photo"]):
        # Delete faces records & purge uploaded photos from disk
        face_rows = conn.execute("SELECT photo_url FROM faces WHERE person_id = ?", (pid,)).fetchall()
        for fr in face_rows:
            purl = fr["photo_url"]
            if purl and purl.startswith("/uploads/"):
                disk_path = UPLOADS_DIR / purl.replace("/uploads/", "")
                if disk_path.exists():
                    try: disk_path.unlink()
                    except Exception: pass
        conn.execute("DELETE FROM faces WHERE person_id = ?", (pid,))
        conn.execute("UPDATE people SET photo = NULL, consent_photo = 0 WHERE id = ?", (pid,))
        audit("purge_face_biometrics", f"person:{pid}:withdrawn_and_shredded")

    if "consent_voice_clone" in b and not int(b["consent_voice_clone"]):
        # Delete voice records & purge audio files from disk
        voice_rows = conn.execute("SELECT audio_url FROM voices WHERE person_id = ?", (pid,)).fetchall()
        for vr in voice_rows:
            aurl = vr["audio_url"]
            if aurl and aurl.startswith("/uploads/"):
                disk_path = UPLOADS_DIR / aurl.replace("/uploads/", "")
                if disk_path.exists():
                    try: disk_path.unlink()
                    except Exception: pass
        conn.execute("UPDATE people SET has_voice = 0, consent_voice_clone = 0, voice_pitch = NULL WHERE id = ?", (pid,))
        conn.execute("DELETE FROM voices WHERE person_id = ?", (pid,))
        conn.execute("UPDATE voice_messages SET active = 0 WHERE person_id = ?", (pid,))
        audit("purge_voice_clone", f"person:{pid}:withdrawn_and_shredded")

    conn.commit()
    audit("update_person", f"person:{pid}:{','.join(keys)}")
    return jsonify(person_public(person_row(pid), True))


@app.route("/api/people/<int:pid>", methods=["DELETE"])
@caregiver_required
def api_delete_person(pid):
    if not person_row(pid):
        return jsonify({"error": "Person not found."}), 404
    conn = db()
    conn.execute("DELETE FROM faces WHERE person_id = ?", (pid,))
    conn.execute("DELETE FROM memory_people WHERE person_id = ?", (pid,))
    conn.execute("DELETE FROM relations WHERE person_id = ? OR related_id = ?", (pid, pid))
    conn.execute("DELETE FROM people WHERE id = ?", (pid,))
    conn.commit()
    audit("delete_person", f"person:{pid}")
    return jsonify({"deleted": pid})


@app.route("/api/people/<int:pid>/photos", methods=["POST"])
@caregiver_required
def api_add_photo(pid):
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404
    if not p["consent_photo"]:
        return jsonify({"error": "No photo consent recorded for this person."}), 422

    photo_file = request.files.get("photo") or request.files.get("file")
    if not photo_file:
        return jsonify({"error": "No file was attached."}), 400

    img_bytes = photo_file.read()
    emb = extract_embedding(img_bytes)
    fname = f"person_{pid}_{now_iso().replace(':', '-')}.png"
    fpath = UPLOADS_DIR / fname
    with open(fpath, "wb") as wf:
        wf.write(img_bytes)
    photo_url = f"/uploads/{fname}"
    conn = db()
    conn.execute(
        "INSERT INTO faces (person_id, embedding, photo_url, created_at) VALUES (?, ?, ?, ?)",
        (pid, json.dumps(emb), photo_url, now_iso())
    )
    if not p["photo"]:
        conn.execute("UPDATE people SET photo = ? WHERE id = ?", (photo_url, pid))
    conn.commit()
    audit("add_photo", f"person:{pid}")
    return jsonify({"enrolled": True}), 201


# ──────────────────────────────────────────────
# /api/graph
# ──────────────────────────────────────────────
@app.route("/api/graph")
def api_graph():
    carer = is_caregiver()
    people = db().execute("SELECT * FROM people").fetchall()
    if not carer:
        people = [p for p in people if p["status"] == "active"]
    pids = set(p["id"] for p in people)

    nodes = [{"id": 0, "name": patient_name(), "relationship": "you", "generation": 0, "patient": True}]
    for p in people:
        mem_count = db().execute("SELECT COUNT(*) as c FROM memory_people WHERE person_id = ?", (p["id"],)).fetchone()["c"]
        nodes.append({
            **person_public(p),
            "generation": generation(p["relationship"]),
            "circle": "friends" if generation(p["relationship"]) is None else "family",
            "memory_count": mem_count,
        })

    edges = [{"from": 0, "to": p["id"], "kind": "to_patient", "label": p["relationship"]} for p in people]
    rels = db().execute("SELECT * FROM relations").fetchall()
    for r in rels:
        if r["person_id"] in pids and r["related_id"] in pids:
            edges.append({"from": r["person_id"], "to": r["related_id"], "kind": r["kind"], "id": r["id"]})

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/relations", methods=["POST"])
@caregiver_required
def api_add_relation():
    b = request.get_json(force=True, silent=True) or {}
    a, bv, kind = b.get("person_id"), b.get("related_id"), b.get("kind")
    if kind not in ("parent", "spouse", "sibling") or not a or not bv or a == bv:
        return jsonify({"error": "Choose two different people and parent, spouse or sibling."}), 400
    cur = db().execute("INSERT INTO relations (person_id, related_id, kind) VALUES (?, ?, ?)", (int(a), int(bv), kind))
    db().commit()
    rid = cur.lastrowid
    audit("add_relation", f"relation:{a}-{kind}-{bv}")
    return jsonify({"id": rid}), 201


@app.route("/api/relations/<int:rid>", methods=["DELETE"])
@caregiver_required
def api_delete_relation(rid):
    db().execute("DELETE FROM relations WHERE id = ?", (rid,))
    db().commit()
    audit("delete_relation", f"relation:{rid}")
    return jsonify({"deleted": rid})


# ──────────────────────────────────────────────
# /api/memories
# ──────────────────────────────────────────────
@app.route("/api/memories")
def api_memories():
    carer = is_caregiver()
    pid_filter = request.args.get("person_id", type=int)
    if pid_filter:
        mem_ids = [r["memory_id"] for r in db().execute("SELECT memory_id FROM memory_people WHERE person_id = ?", (pid_filter,)).fetchall()]
        if mem_ids:
            placeholders = ",".join("?" * len(mem_ids))
            rows = db().execute(f"SELECT * FROM memories WHERE id IN ({placeholders}) ORDER BY COALESCE(year, 9999), id", mem_ids).fetchall()
        else:
            rows = []
    else:
        rows = db().execute("SELECT * FROM memories ORDER BY COALESCE(year, 9999), id").fetchall()

    result = []
    for r in rows:
        m = dict(r)
        ppl_ids = memory_people_ids(m["id"])
        if not carer:
            people_data = [person_row(pid) for pid in ppl_ids]
            if m["privacy"] != "normal" or any(p and p["status"] != "active" for p in people_data):
                continue
        result.append(mem_public(r, carer))
    return jsonify(result)


@app.route("/api/memories/<int:mid>")
def api_memory_detail(mid):
    carer = is_caregiver()
    row = db().execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    if not row:
        return jsonify({"error": "That memory isn't available."}), 404
    o = mem_public(row, carer)

    # Related memories
    people_data = load_all_people()
    places_data = load_all_places()
    people_map = {p["id"]: p for p in people_data}
    places_map = {pl["id"]: pl for pl in places_data}
    all_mems = load_all_memories()
    search_q = f"{row['title']} {row['content']}"
    hits = search_memories(search_q, all_mems, people_map, places_map, exclude_ids={mid}, k=4)
    o["related"] = [{"id": h["memory"]["id"], "title": h["memory"]["title"], "year": h["memory"].get("year")} for h in hits[:3]]
    return jsonify(o)


@app.route("/api/memories", methods=["POST"])
@caregiver_required
def api_create_memory():
    b = request.get_json(force=True, silent=True) or {}
    title = (b.get("title") or "").strip()
    content = (b.get("content") or "").strip()
    if not title or not content:
        return jsonify({"error": "A memory needs a title and a few words about it."}), 400

    privacy = b.get("privacy", setting("default_privacy"))
    if privacy not in PRIVACY_LEVELS:
        return jsonify({"error": "Privacy must be normal, private, sensitive or locked."}), 400

    year = None
    if "year" in b and b["year"] not in ("", None):
        try:
            year = int(b["year"])
            if year < 1900 or year > 2100:
                return jsonify({"error": "Year must be a number like 1998."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Year must be a number like 1998."}), 400

    place_id = None
    if b.get("place"):
        pl_name = b["place"].strip()
        pl = db().execute("SELECT id FROM places WHERE LOWER(name)=LOWER(?)", (pl_name,)).fetchone()
        if pl:
            place_id = pl["id"]
        else:
            cur = db().execute("INSERT INTO places (name, region) VALUES (?, '')", (pl_name,))
            db().commit()
            place_id = cur.lastrowid

    conn = db()
    cur = conn.execute(
        """INSERT INTO memories (title, content, memory_date, year, life_stage, category,
        place_id, privacy, song_title, song_artist, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'caregiver', ?)""",
        (title, content, b.get("memory_date", ""), year,
         b.get("life_stage", "recent"), b.get("category", "family"),
         place_id, privacy, b.get("song_title", ""), b.get("song_artist", ""), now_iso())
    )
    mid = cur.lastrowid

    for pid in b.get("people", []):
        conn.execute("INSERT OR IGNORE INTO memory_people (memory_id, person_id) VALUES (?, ?)", (mid, int(pid)))
    conn.commit()
    audit("create_memory", f"memory:{mid}")

    row = db().execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    return jsonify(mem_public(row, True, True)), 201


@app.route("/api/memories/<int:mid>", methods=["PATCH"])
@caregiver_required
def api_update_memory(mid):
    row = db().execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    if not row:
        return jsonify({"error": "Memory not found."}), 404
    b = request.get_json(force=True, silent=True) or {}

    updates = {}
    for k in ("title", "content", "memory_date", "song_title", "song_artist"):
        if k in b:
            v = str(b[k] or "").strip()
            if k in ("title", "content") and not v:
                return jsonify({"error": "Title and description can't be empty."}), 400
            updates[k] = v
    if "privacy" in b:
        if b["privacy"] not in PRIVACY_LEVELS:
            return jsonify({"error": "Privacy must be normal, private, sensitive or locked."}), 400
        updates["privacy"] = b["privacy"]
    if "year" in b:
        if b["year"] in ("", None):
            updates["year"] = None
        else:
            try:
                y = int(b["year"])
                if y < 1900 or y > 2100:
                    return jsonify({"error": "Year must be a number like 1998."}), 400
                updates["year"] = y
            except (ValueError, TypeError):
                return jsonify({"error": "Year must be a number like 1998."}), 400
    for k in ("category", "life_stage"):
        if k in b:
            updates[k] = b[k]

    if not updates and "people" not in b:
        return jsonify({"error": "Nothing to update."}), 400

    conn = db()
    for k, v in updates.items():
        conn.execute(f"UPDATE memories SET {k} = ? WHERE id = ?", (v, mid))

    if "people" in b:
        conn.execute("DELETE FROM memory_people WHERE memory_id = ?", (mid,))
        for pid in b["people"]:
            if not person_row(int(pid)):
                return jsonify({"error": "One of the people wasn't found."}), 404
            conn.execute("INSERT INTO memory_people (memory_id, person_id) VALUES (?, ?)", (mid, int(pid)))

    conn.commit()
    audit("update_memory", f"memory:{mid}:{','.join(updates.keys())}")
    return jsonify(mem_public(db().execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone(), True))


@app.route("/api/memories/<int:mid>/unlock", methods=["POST"])
@caregiver_required
def api_unlock_memory(mid):
    row = db().execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    if not row:
        return jsonify({"error": "Memory not found."}), 404
    audit("unlock_memory", f"memory:{mid}")
    return jsonify(mem_public(row, True, True))


@app.route("/api/memories/<int:mid>", methods=["DELETE"])
@caregiver_required
def api_delete_memory(mid):
    if not db().execute("SELECT id FROM memories WHERE id = ?", (mid,)).fetchone():
        return jsonify({"error": "Memory not found."}), 404
    conn = db()
    conn.execute("DELETE FROM memory_people WHERE memory_id = ?", (mid,))
    conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
    conn.commit()
    audit("delete_memory", f"memory:{mid}")
    return jsonify({"deleted": mid})


@app.route("/api/memories/<int:mid>/photo", methods=["POST"])
@caregiver_required
def api_attach_memory_photo(mid):
    row = db().execute("SELECT id FROM memories WHERE id = ?", (mid,)).fetchone()
    if not row:
        return jsonify({"error": "Memory not found."}), 404
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file was attached."}), 400
    fname = f"memory_{mid}_photo.png"
    fpath = UPLOADS_DIR / fname
    f.save(str(fpath))
    db().execute("UPDATE memories SET photo_url = ? WHERE id = ?", (f"/uploads/{fname}", mid))
    db().commit()
    audit("attach_memory_photo", f"memory:{mid}")
    return jsonify({"attached": True}), 201


@app.route("/api/memories/<int:mid>/song", methods=["POST"])
@caregiver_required
def api_attach_memory_song(mid):
    row = db().execute("SELECT id FROM memories WHERE id = ?", (mid,)).fetchone()
    if not row:
        return jsonify({"error": "Memory not found."}), 404
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file was attached."}), 400
    fname = f"memory_{mid}_song{Path(f.filename).suffix}"
    fpath = UPLOADS_DIR / fname
    f.save(str(fpath))
    db().execute("UPDATE memories SET song_url = ? WHERE id = ?", (f"/uploads/{fname}", mid))
    db().commit()
    audit("attach_song", f"memory:{mid}")
    return jsonify({"attached": True}), 201


# ──────────────────────────────────────────────
# /api/places
# ──────────────────────────────────────────────
@app.route("/api/places")
def api_places():
    carer = is_caregiver()
    rows = db().execute("SELECT * FROM places").fetchall()
    result = []
    for pl in rows:
        pl = dict(pl)
        mems = db().execute(
            "SELECT m.* FROM memories m JOIN memory_people mp ON m.id=mp.memory_id WHERE m.place_id = ?", (pl["id"],)
        ).fetchall()
        if not carer:
            mems = [m for m in mems if m["privacy"] == "normal"]
        ppl_ids = set()
        for m in mems:
            for pid in memory_people_ids(m["id"]):
                ppl_ids.add(pid)
        ppl = []
        for pid in ppl_ids:
            pr = person_row(pid)
            if pr and (carer or pr["status"] == "active"):
                ppl.append({"id": pr["id"], "name": pr["name"]})
        if not carer and not mems:
            continue
        result.append({
            **pl,
            "memory_count": len(mems),
            "photo_count": sum(1 for m in mems if m["photo_url"]),
            "people": ppl,
            "years": sorted(set(m["year"] for m in mems if m["year"])),
            "memory_ids": [m["id"] for m in mems],
        })
    return jsonify(result)


@app.route("/api/places", methods=["POST"])
@caregiver_required
def api_create_place():
    b = request.get_json(force=True, silent=True) or {}
    name = (b.get("name") or "").strip()
    if not name:
        return jsonify({"error": "A place needs a name."}), 400
    if db().execute("SELECT id FROM places WHERE LOWER(name) = LOWER(?)", (name,)).fetchone():
        return jsonify({"error": "That place already exists."}), 400

    lat = b.get("lat")
    lng = b.get("lng")
    if lat in ("", None): lat = None
    else: lat = float(lat)
    if lng in ("", None): lng = None
    else: lng = float(lng)

    cur = db().execute("INSERT INTO places (name, region, lat, lng, note) VALUES (?, ?, ?, ?, ?)",
                       (name, b.get("region", ""), lat, lng, b.get("note", "")))
    db().commit()
    audit("create_place", f"place:{cur.lastrowid}")
    pl = db().execute("SELECT * FROM places WHERE id = ?", (cur.lastrowid,)).fetchone()
    return jsonify(dict(pl)), 201


# ──────────────────────────────────────────────
# /api/recognize, /api/chat, /api/suggest, /api/teach
# ──────────────────────────────────────────────
@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    carer = is_caregiver()
    photo_file = request.files.get("image") or request.files.get("file")
    if not photo_file:
        return jsonify({"error": "No file was attached."}), 400

    img_bytes = photo_file.read()

    # Pre-recognition Image Quality Gate (Blur & Lighting Verification)
    quality = check_image_quality(img_bytes)

    query_emb = extract_embedding(img_bytes)
    enrolled = get_enrolled_faces()
    threshold = float(setting("face_recognize_threshold") or 0.90)
    result = match_face(query_emb, enrolled, recognize_threshold=threshold)

    carer_name = get_carer_display()
    preview = request.args.get("preview") == "1" and carer

    # If photo quality is poor, enforce safety refusal rather than guessing
    if not quality["ok"]:
        out = {
            "status": "uncertain",
            "confidence": result["score"],
            "confidence_band": "low",
            "basic_mode": False,
            "person": None,
            "sources": [],
            "why": [{"type": "rule", "text": quality["problem"]}],
            "confidence_label": "Poor Image Quality",
            "text": f"I can't see the face clearly ({quality['problem'].lower()}). Let's ask {carer_name} or try a clearer photo.",
            "quality": quality,
            "preview": preview,
        }
        if not preview:
            db().execute(
                "INSERT INTO interactions (kind, person_id, outcome, score, summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("recognize", None, "quality_fail", result["score"], out["text"], now_iso())
            )
            db().commit()
        return jsonify(out)

    out = {
        "status": result["status"],
        "confidence": result["score"],
        "margin": result.get("margin", 0.0),
        "confidence_band": result.get("confidence_band", "none"),
        "basic_mode": False,
        "person": None,
        "sources": [],
        "why": [],
        "confidence_label": "None",
        "quality": quality,
        "preview": preview,
    }

    if result["status"] == "recognized" and result["person_id"]:
        p = person_row(result["person_id"])
        if p and p["status"] != "active":
            out.update({
                "status": "protected",
                "text": f"This is someone very dear to your family. {carer_name} can sit with you and tell you about them.",
            })
            if not preview:
                add_alert("info", "", "A protected person's photo was shown", p["name"])
        elif p:
            band_label = "High" if result.get("confidence_band") == "high" else "Medium"
            out.update({
                "status": "recognized",
                "person": person_public(p),
                "text": f"This is {display_name(p['name'])}, your {clean_rel(p['relationship'])}.",
                "confidence_label": band_label,
                "why": [
                    {"type": "face", "text": f"The face matched {p['name']}'s saved photos (score {result['score']:.2f}, safety margin +{result.get('margin', 0.0):.2f})"},
                    {"type": "rule", "text": f"Confidence band: {band_label}. Exceeded safety threshold of {threshold:.2f}."},
                ] + ([{"type": "rule", "text": note} for note in result.get("safety_notes", [])]),
            })
    elif result["status"] == "uncertain":
        reason_text = "The closest match was below the naming threshold"
        if result.get("confidence_band") == "ambiguous":
            reason_text = "Lookalike protection: match was too close to another family member to name without guessing"

        out.update({
            "text": f"I can't tell for certain who this is. {carer_name} can help us.",
            "confidence_label": "Uncertain (Refused Guess)",
            "why": [
                {"type": "rule", "text": reason_text},
                {"type": "rule", "text": "EchoBond never guesses below the safety threshold to prevent confusion."}
            ],
        })
        if carer and result["person_id"]:
            pr = person_row(result["person_id"])
            if pr:
                out["caregiver_hint"] = f"Closest match: {pr['name']} (score {result['score']:.2f}, margin {result.get('margin', 0.0):.2f}). Add more photos to improve."
    else:
        out.update({
            "text": f"I'm not sure who this is, and that's alright. Let's ask {carer_name} together.",
            "can_teach": True,
            "why": [{"type": "rule", "text": "The face didn't match anyone EchoBond knows (score below 0.75)"}],
        })

    if not preview:
        db().execute(
            "INSERT INTO interactions (kind, person_id, outcome, score, summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("recognize", result.get("person_id"), out["status"], result["score"], out.get("text", ""), now_iso())
        )
        db().commit()

    return jsonify(out)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    carer = is_caregiver()
    b = request.get_json(force=True, silent=True) or {}
    msg = (b.get("message") or "").strip()[:500]
    if not msg:
        return jsonify({"error": "Please type or say something."}), 400

    session_key = b.get("session", "")
    preview = bool(b.get("preview")) and carer
    norm = normalize_question(msg)

    people = load_all_people()
    memories = load_all_memories()
    places = load_all_places()
    reminders = load_all_reminders()
    schedules = load_all_schedules()

    ctx = sessions.get(session_key, {"person_ids": [], "memory_ids": [], "told_ids": []})
    engine = ConversationEngine(db(), patient_name())
    result, alert_data, new_ctx = engine.answer_query(msg, ctx, people, memories, places, reminders, schedules)

    if new_ctx:
        old_told = ctx.get("told_ids", [])
        new_ctx.setdefault("told_ids", [])
        new_ctx["told_ids"] = list(set(old_told + new_ctx.get("memory_ids", [])))
        sessions[session_key] = new_ctx

    threshold = int(setting("repeat_threshold"))
    window = int(setting("repeat_window_hours"))
    cutoff = (datetime.utcnow() - timedelta(hours=window)).isoformat() + "Z"

    if preview:
        count = db().execute(
            "SELECT COUNT(*) as c FROM interactions WHERE kind='chat' AND normalized=? AND created_at>=?",
            (norm, cutoff)
        ).fetchone()["c"] + 1
        result["preview"] = True
        result["repeat"] = {"count": count, "threshold": threshold, "would_alert": count >= threshold}
        return jsonify(result)

    if alert_data:
        add_alert(alert_data["level"], alert_data["kind"], alert_data["reason"], alert_data["detail"], alert_data.get("ref", ""))

    db().execute(
        "INSERT INTO interactions (kind, outcome, question, normalized, category, memory_ids, session, summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("chat", result.get("outcome", "answered"), msg, norm, result.get("category", ""),
         json.dumps(result.get("memory_ids", [])), session_key, f"Q: {msg} | A: {result['text'][:100]}", now_iso())
    )
    db().commit()

    count = db().execute(
        "SELECT COUNT(*) as c FROM interactions WHERE kind='chat' AND normalized=? AND created_at>=?",
        (norm, cutoff)
    ).fetchone()["c"]

    already = db().execute(
        "SELECT id FROM alerts WHERE kind='repeated_question' AND ref=? AND created_at>=?",
        (norm, datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat() + "Z")
    ).fetchone()

    if count >= threshold and not already and result.get("outcome") != "safety":
        add_alert("info", "repeated_question", "Repeated question",
                  f'Asked {count} times in {window} hours: "{msg}"', norm)

    result["repeat"] = {"count": count}
    return jsonify(result)


@app.route("/api/suggest", methods=["POST"])
@caregiver_required
def api_suggest():
    photo_file = request.files.get("image") or request.files.get("file")
    if not photo_file:
        return jsonify({"error": "No file was attached."}), 400

    img_bytes = photo_file.read()
    query_emb = extract_embedding(img_bytes)
    enrolled = get_enrolled_faces()
    threshold = float(setting("face_recognize_threshold") or 0.90)
    result = match_face(query_emb, enrolled, recognize_threshold=threshold)

    ppl = []
    if result["status"] == "recognized" and result["person_id"]:
        p = person_row(result["person_id"])
        if p:
            ppl.append({"id": p["id"], "name": p["name"], "relationship": p["relationship"],
                        "protected": p["status"] != "active", "score": result["score"]})

    active_names = [display_name(p["name"]) for p in ppl if not p.get("protected")]
    return jsonify({
        "saved": False,
        "people": ppl,
        "unknown_faces": 0 if ppl else 1,
        "caption": None,
        "caption_model": "unavailable",
        "possible_event": None,
        "draft": {
            "title": f"Photo with {' and '.join(active_names)}" if active_names else "Family photo",
            "content": f"A photo with {' and '.join(active_names)}." if active_names else "",
            "people": [p["id"] for p in ppl if not p.get("protected")],
            "privacy": "normal" if all(not p.get("protected") for p in ppl) else "sensitive",
        },
        "basic_mode": False,
        "note": "Server-side face recognition with PIL + numpy.",
    })


@app.route("/api/teach", methods=["POST"])
@caregiver_required
def api_teach():
    f = request.form
    name = (f.get("name") or "").strip()
    rel_txt = (f.get("relationship") or "").strip()
    if not name or not rel_txt:
        return jsonify({"error": "Add their name and how they're related."}), 400
    if f.get("consent_photo") != "1":
        return jsonify({"error": "Teaching EchoBond a face needs that person's consent."}), 400
    by = (f.get("consent_given_by") or "").strip()
    if not by:
        return jsonify({"error": "Record who gave consent."}), 400

    photo_file = request.files.get("image") or request.files.get("file")
    if not photo_file:
        return jsonify({"error": "Attach the photo of the person to teach."}), 400

    img_bytes = photo_file.read()
    emb = extract_embedding(img_bytes)

    conn = db()
    cur = conn.execute(
        """INSERT INTO people (key, name, relationship, nickname, description, status,
        consent_photo, consent_voice_clone, consent_given_by, consent_date,
        avatar_color, photo, has_voice, created_at)
        VALUES ('', ?, ?, ?, ?, 'active', 1, 0, ?, ?, '#c9a36b', NULL, 0, ?)""",
        (name, rel_txt, f.get("nickname", ""), f.get("description", ""), by, now_iso(), now_iso())
    )
    pid = cur.lastrowid

    fname = f"person_{pid}_teach.png"
    fpath = UPLOADS_DIR / fname
    with open(fpath, "wb") as wf:
        wf.write(img_bytes)
    photo_url = f"/uploads/{fname}"

    conn.execute("UPDATE people SET photo = ? WHERE id = ?", (photo_url, pid))
    conn.execute(
        "INSERT INTO faces (person_id, embedding, photo_url, created_at) VALUES (?, ?, ?, ?)",
        (pid, json.dumps(emb), photo_url, now_iso())
    )

    memory_id = None
    if (f.get("memory_title") or "").strip() and (f.get("memory_content") or "").strip():
        cur2 = conn.execute(
            """INSERT INTO memories (title, content, life_stage, category, privacy, source, created_at)
            VALUES (?, ?, 'recent', 'family', 'normal', 'teach', ?)""",
            (f["memory_title"].strip(), f["memory_content"].strip(), now_iso())
        )
        memory_id = cur2.lastrowid
        conn.execute("INSERT INTO memory_people (memory_id, person_id) VALUES (?, ?)", (memory_id, pid))

    conn.commit()
    audit("teach_person", f"person:{pid}")

    p = person_row(pid)
    return jsonify({"person": person_public(p, True), "photos": [{"enrolled": True}], "memory_id": memory_id}), 201


# ──────────────────────────────────────────────
# /api/context, /api/reminders, /api/notes
# ──────────────────────────────────────────────
@app.route("/api/context")
def api_context():
    d = datetime.now()
    h = d.hour
    pod = "morning" if h < 12 else "afternoon" if h < 17 else "evening" if h < 21 else "night"
    return jsonify({
        "greeting": f"Good {'evening' if pod == 'night' else pod}, {patient_name()}.",
        "orientation": f"Today is {d.strftime('%A')}, {d.day} {d.strftime('%B')}.",
        "part_of_day": pod,
        "calm_mode": "Evening mode: slower, shorter prompts." if pod in ("evening", "night") else None,
    })


@app.route("/api/reminders")
def api_reminders():
    rows = db().execute("SELECT * FROM reminders ORDER BY time_of_day").fetchall()
    result = []
    for r in rows:
        r = dict(r)
        done = db().execute(
            "SELECT id FROM reminder_log WHERE reminder_id = ? AND created_at >= ?",
            (r["id"], datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat() + "Z")
        ).fetchone()
        kind = r.get("kind", "routine")
        carer_name = get_carer_display()
        phrase = f"It's time for your medicine. {carer_name} will help you." if kind == "medication" else r["text"]
        r["phrase"] = phrase
        r["done_today"] = done is not None
        result.append(r)
    return jsonify(result)


@app.route("/api/reminders", methods=["POST"])
@caregiver_required
def api_create_reminder():
    b = request.get_json(force=True, silent=True) or {}
    text = (b.get("text") or "").strip()
    tm = b.get("time_of_day", "")
    kind = b.get("kind", "routine")
    if not text or not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", tm):
        return jsonify({"error": "A reminder needs text and a time like 09:30."}), 400
    if kind not in ("routine", "visit", "appointment", "medication"):
        return jsonify({"error": "Type must be routine, visit, appointment or medication."}), 400
    if kind == "medication" and re.search(r"\d+\s?(mg|ml|mcg|tablets?|pills?|units?)\b", text, re.IGNORECASE):
        return jsonify({"error": "Please don't put doses in reminders."}), 400

    cur = db().execute("INSERT INTO reminders (text, time_of_day, kind, days, active) VALUES (?, ?, ?, 'daily', 1)", (text, tm, kind))
    db().commit()
    rid = cur.lastrowid
    audit("create_reminder", f"reminder:{rid}")
    return jsonify(dict(db().execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone())), 201


@app.route("/api/reminders/<int:rid>", methods=["PATCH"])
@caregiver_required
def api_update_reminder(rid):
    r = db().execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone()
    if not r:
        return jsonify({"error": "Reminder not found."}), 404
    b = request.get_json(force=True, silent=True) or {}
    if "active" in b:
        db().execute("UPDATE reminders SET active = ? WHERE id = ?", (1 if b["active"] else 0, rid))
    if "text" in b:
        db().execute("UPDATE reminders SET text = ? WHERE id = ?", (b["text"], rid))
    if "time_of_day" in b:
        db().execute("UPDATE reminders SET time_of_day = ? WHERE id = ?", (b["time_of_day"], rid))
    if "kind" in b:
        db().execute("UPDATE reminders SET kind = ? WHERE id = ?", (b["kind"], rid))
    db().commit()
    audit("update_reminder", f"reminder:{rid}")
    return jsonify(dict(db().execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone()))


@app.route("/api/reminders/<int:rid>", methods=["DELETE"])
@caregiver_required
def api_delete_reminder(rid):
    db().execute("DELETE FROM reminders WHERE id = ?", (rid,))
    db().commit()
    audit("delete_reminder", f"reminder:{rid}")
    return jsonify({"deleted": rid})


@app.route("/api/reminders/<int:rid>/done", methods=["POST"])
def api_reminder_done(rid):
    if not db().execute("SELECT id FROM reminders WHERE id = ?", (rid,)).fetchone():
        return jsonify({"error": "Reminder not found."}), 404
    db().execute("INSERT INTO reminder_log (reminder_id, status, created_at) VALUES (?, 'done', ?)", (rid, now_iso()))
    db().commit()
    return jsonify({"done": rid})


@app.route("/api/notes")
@caregiver_required
def api_notes():
    rows = db().execute("SELECT * FROM notes ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/notes", methods=["POST"])
@caregiver_required
def api_create_note():
    b = request.get_json(force=True, silent=True) or {}
    if b.get("mood") not in ("good", "okay", "hard"):
        return jsonify({"error": "Choose how the day went: good, okay or hard."}), 400
    cur = db().execute(
        "INSERT INTO notes (mood, note, author, created_at) VALUES (?, ?, ?, ?)",
        (b["mood"], (b.get("note") or "")[:600], b.get("author", "caregiver"), now_iso())
    )
    db().commit()
    nid = cur.lastrowid
    audit("add_note", f"note:{nid}")
    return jsonify(dict(db().execute("SELECT * FROM notes WHERE id = ?", (nid,)).fetchone())), 201


# ──────────────────────────────────────────────
# /api/alerts, /api/interactions, /api/stats, /api/summary
# ──────────────────────────────────────────────
@app.route("/api/alerts")
@caregiver_required
def api_alerts():
    rows = db().execute("SELECT * FROM alerts ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts/<int:aid>/ack", methods=["POST"])
@caregiver_required
def api_ack_alert(aid):
    a = db().execute("SELECT * FROM alerts WHERE id = ?", (aid,)).fetchone()
    if not a:
        return jsonify({"error": "Alert not found."}), 404
    db().execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (aid,))
    db().commit()
    audit("ack_alert", f"alert:{aid}")
    return jsonify({"acknowledged": aid})


@app.route("/api/interactions")
@caregiver_required
def api_interactions():
    days = int(request.args.get("days", 7))
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    rows = db().execute("SELECT * FROM interactions WHERE created_at >= ? ORDER BY created_at DESC", (cutoff,)).fetchall()

    result = []
    for r in rows:
        r = dict(r)
        # Apply filters
        for fk in ("normalized", "session", "kind", "outcome"):
            fv = request.args.get(fk)
            if fv and r.get(fk) != fv:
                break
        else:
            pr = person_row(r.get("person_id")) if r.get("person_id") else None
            r["person_name"] = pr["name"] if pr else None
            result.append(r)
    return jsonify(result)


@app.route("/api/stats")
@caregiver_required
def api_stats():
    day_cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"
    week_cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"

    recent = db().execute("SELECT * FROM interactions WHERE created_at >= ?", (day_cutoff,)).fetchall()
    week = db().execute("SELECT * FROM interactions WHERE kind='recognize' AND created_at >= ?", (week_cutoff,)).fetchall()

    by_day = {}
    for i in week:
        k = f"{i['created_at'][:10]}|{i['outcome']}"
        by_day[k] = by_day.get(k, 0) + 1

    pc = {}
    for m in db().execute("SELECT privacy, COUNT(*) as c FROM memories GROUP BY privacy").fetchall():
        pc[m["privacy"]] = m["c"]

    people_count = db().execute("SELECT COUNT(*) as c FROM people").fetchone()["c"]
    mem_count = db().execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    sens_count = db().execute("SELECT COUNT(*) as c FROM memories WHERE privacy IN ('sensitive','locked')").fetchone()["c"]

    people_needing = []
    for p in db().execute("SELECT id, name FROM people WHERE status='active' AND consent_photo=1").fetchall():
        fc = db().execute("SELECT COUNT(*) as c FROM faces WHERE person_id = ?", (p["id"],)).fetchone()["c"]
        if fc < 2:
            people_needing.append({"id": p["id"], "name": p["name"], "samples": fc})

    return jsonify({
        "open_alerts": db().execute("SELECT COUNT(*) as c FROM alerts WHERE acknowledged=0").fetchone()["c"],
        "urgent_alerts": db().execute("SELECT COUNT(*) as c FROM alerts WHERE acknowledged=0 AND level='urgent'").fetchone()["c"],
        "recognitions_24h": sum(1 for i in recent if i["kind"] == "recognize"),
        "not_named_24h": sum(1 for i in recent if i["kind"] == "recognize" and i["outcome"] in ("uncertain", "unknown", "no_face")),
        "chats_24h": sum(1 for i in recent if i["kind"] == "chat"),
        "no_memory_24h": sum(1 for i in recent if i["kind"] == "chat" and i["outcome"] == "no_memory"),
        "memories": mem_count,
        "sensitive_memories": sens_count,
        "privacy_counts": pc,
        "people": people_count,
        "recognition_by_day": [{"day": k.split("|")[0], "outcome": k.split("|")[1], "n": v} for k, v in by_day.items()],
        "people_needing_photos": people_needing,
    })


@app.route("/api/summary")
@caregiver_required
def api_summary():
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat() + "Z"
    today = db().execute("SELECT * FROM interactions WHERE created_at >= ?", (today_start,)).fetchall()

    ppl_names = list(set(
        person_row(i["person_id"])["name"]
        for i in today if i["kind"] == "recognize" and i["outcome"] == "recognized" and i["person_id"] and person_row(i["person_id"])
    ))
    unc = sum(1 for i in today if i["kind"] == "recognize" and i["outcome"] in ("uncertain", "unknown", "no_face"))
    chats = sum(1 for i in today if i["kind"] == "chat")
    new_m = db().execute("SELECT COUNT(*) as c FROM memories WHERE source != 'demo' AND created_at >= ?", (today_start,)).fetchone()["c"]
    done = db().execute("SELECT COUNT(DISTINCT reminder_id) as c FROM reminder_log WHERE created_at >= ?", (today_start,)).fetchone()["c"]
    total = db().execute("SELECT COUNT(*) as c FROM reminders WHERE active=1").fetchone()["c"]
    open_alerts = db().execute("SELECT COUNT(*) as c FROM alerts WHERE acknowledged=0").fetchone()["c"]

    def plural(n, s, p=None):
        return f"{n} {s if n == 1 else (p or s + 's')}"

    lines = [
        f"{plural(len(ppl_names), 'person', 'people')} recognised{' (' + ', '.join(n.split()[0] for n in ppl_names) + ')' if ppl_names else ''}",
        f"{plural(unc, 'face')} EchoBond wasn't sure about",
        plural(chats, "conversation"),
        f"{plural(new_m, 'new memory', 'new memories')} added",
        f"{done} of {plural(total, 'reminder')} marked done",
        f"{plural(open_alerts, 'alert')} {'needs' if open_alerts == 1 else 'need'} attention",
    ]
    return jsonify({
        "date": datetime.now().strftime("%A, %d %B %Y"),
        "people_recognized": ppl_names,
        "uncertain": unc,
        "conversations": chats,
        "new_memories": new_m,
        "reminders_completed": done,
        "reminders_total": total,
        "alerts_open": open_alerts,
        "lines": lines,
    })


@app.route("/api/repeated")
@caregiver_required
def api_repeated():
    cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"
    rows = db().execute(
        "SELECT normalized, question, COUNT(*) as n, MAX(created_at) as last FROM interactions "
        "WHERE kind='chat' AND normalized IS NOT NULL AND normalized != '' AND created_at >= ? "
        "GROUP BY normalized HAVING n >= 2 ORDER BY n DESC", (cutoff,)
    ).fetchall()
    return jsonify([{"normalized": r["normalized"], "n": r["n"], "example": r["question"], "last": r["last"]} for r in rows])


# ──────────────────────────────────────────────
# /api/assistant
# ──────────────────────────────────────────────
@app.route("/api/assistant", methods=["POST"])
@caregiver_required
def api_assistant():
    b = request.get_json(force=True, silent=True) or {}
    t = (b.get("question") or "").lower().strip()
    if not t:
        return jsonify({"error": "Ask a question about today, alerts, repeated questions or memories to add."}), 400

    def plural(n, s, p=None):
        return f"{n} {s if n == 1 else (p or s + 's')}"

    if re.search(r"\b(alert|alerts|attention|urgent|worry|worried)\b", t):
        alerts = [dict(a) for a in db().execute("SELECT * FROM alerts WHERE acknowledged=0").fetchall()]
        return jsonify({
            "intent": "alerts",
            "answer": f"{plural(len(alerts), 'alert')} {'needs' if len(alerts) == 1 else 'need'} attention." if alerts else "No alerts need your attention.",
            "items": [f"{a['reason']}: \"{a['detail']}\"" for a in alerts],
            "based_on": "Open alerts",
        })

    if re.search(r"\b(repeat|repeated|again|same question|keeps asking)\b", t):
        cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"
        rows = db().execute(
            "SELECT normalized, question, COUNT(*) as n FROM interactions "
            "WHERE kind='chat' AND normalized IS NOT NULL AND created_at >= ? "
            "GROUP BY normalized HAVING n >= 2 ORDER BY n DESC", (cutoff,)
        ).fetchall()
        return jsonify({
            "intent": "repeated",
            "answer": f"{plural(len(rows), 'question')} came up more than once in the last 24 hours." if rows else "No question was asked more than once in the last 24 hours.",
            "items": [f"\"{r['question']}\": {r['n']} times" for r in rows],
            "based_on": "Conversations in the last 24 hours",
        })

    if re.search(r"\b(today|happened|summary|how was|overview|day)\b", t):
        from flask import redirect
        # Reuse summary
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat() + "Z"
        today = db().execute("SELECT * FROM interactions WHERE created_at >= ?", (today_start,)).fetchall()
        chats = sum(1 for i in today if i["kind"] == "chat")
        recog = sum(1 for i in today if i["kind"] == "recognize")
        return jsonify({
            "intent": "today",
            "answer": "Here's today so far.",
            "items": [f"{recog} recognitions", f"{chats} conversations"],
            "based_on": "Today's activity log",
        })

    return jsonify({
        "intent": "help",
        "answer": "I can answer questions about EchoBond's own records. Try one of these:",
        "items": ["What happened today?", "Were any questions repeated?", "What alerts need attention?"],
        "based_on": "Nothing (help)",
    })


# ──────────────────────────────────────────────
# /api/audit, /api/privacy, /api/settings, /api/export, /api/erase-all
# ──────────────────────────────────────────────
@app.route("/api/audit")
@caregiver_required
def api_audit():
    rows = db().execute("SELECT * FROM audit ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/privacy")
@caregiver_required
def api_privacy():
    people = db().execute("SELECT * FROM people").fetchall()
    return jsonify({
        "people": len(people),
        "face_consent": sum(1 for p in people if p["consent_photo"]),
        "voice_consent": sum(1 for p in people if p["consent_voice_clone"]),
        "photos": db().execute("SELECT COUNT(*) as c FROM faces").fetchone()["c"],
        "voice_samples": sum(1 for p in people if p["has_voice"]),
        "face_embeddings": db().execute("SELECT COUNT(*) as c FROM faces").fetchone()["c"],
        "memories": db().execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"],
        "interactions": db().execute("SELECT COUNT(*) as c FROM interactions").fetchone()["c"],
        "camera_frames_stored": 0,
        "encryption_at_rest": False,
        "principles": {
            "purpose_limitation": "Local patient recognition & wellbeing connection only. Zero 3rd-party data transfer.",
            "data_minimization": "Live camera frames are processed in-memory and discarded. Raw live video is never stored.",
            "storage_locality": "100% local on-premises SQLite database. No external cloud dependencies for memory store.",
            "right_to_erasure": "Withdrawing face or voice consent immediately shreds all embeddings, photos, and voice WAV files.",
            "anti_hallucination": "Every conversational memory answer is strictly verified against caregiver-entered archives.",
            "recognition_safety": "Dual confidence threshold (>= 0.90) and lookalike margin rule (>= 0.045) prevent misidentifying strangers or relatives."
        },
        "consent_records": [{
            "id": p["id"], "name": p["name"],
            "relationship": p["relationship"],
            "consent_photo": bool(p["consent_photo"]),
            "consent_voice_clone": bool(p["consent_voice_clone"]),
            "consent_given_by": p["consent_given_by"] or "Not recorded",
            "consent_date": p["consent_date"] or "Not recorded",
            "status": p["status"],
            "biometric_erasure_supported": True
        } for p in people],
    })


@app.route("/api/settings")
def api_settings():
    threshold = float(setting("face_recognize_threshold") or 0.90)
    return jsonify({
        "face_recognize_threshold": setting("face_recognize_threshold"),
        "voice_lang": setting("voice_lang"),
        "voice_rate": setting("voice_rate"),
        "repeat_window_hours": setting("repeat_window_hours"),
        "repeat_threshold": setting("repeat_threshold"),
        "default_privacy": setting("default_privacy"),
        "face_thresholds_in_use": {"recognize": threshold, "uncertain": 0.75, "minimum": 0.88},
        "patient_name": patient_name(),
        "has_gemini_key": bool(get_gemini_api_key(db())),
        "gemini_api_key": setting("gemini_api_key") or "",
    })


@app.route("/api/settings", methods=["PATCH"])
@caregiver_required
def api_update_settings():
    b = request.get_json(force=True, silent=True) or {}
    conn = db()
    for k, v in b.items():
        if k == "face_recognize_threshold":
            if v in ("default", None, ""):
                conn.execute("DELETE FROM settings WHERE key = ?", (k,))
            else:
                x = float(v)
                if not (0.88 <= x <= 0.99):
                    return jsonify({"error": "The face threshold must be between 0.88 and 0.99."}), 400
                conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, f"{x:.2f}"))
        elif k == "voice_lang":
            if v not in ("en-IN", "en-GB", "en-US"):
                return jsonify({"error": "Voice language must be English (India), English (UK) or English (US)."}), 400
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, v))
        elif k == "voice_rate":
            x = float(v)
            if not (0.6 <= x <= 1.1):
                return jsonify({"error": "Speaking speed must be between 0.6 and 1.1."}), 400
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, f"{x:.2f}"))
        elif k == "repeat_window_hours":
            x = int(v)
            if not (1 <= x <= 72):
                return jsonify({"error": "The repeated-question window must be 1 to 72 hours."}), 400
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, str(x)))
        elif k == "repeat_threshold":
            x = int(v)
            if not (2 <= x <= 10):
                return jsonify({"error": "Repeated-question count must be 2 to 10."}), 400
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, str(x)))
        elif k == "default_privacy":
            if v not in PRIVACY_LEVELS:
                return jsonify({"error": "Default privacy must be normal, private, sensitive or locked."}), 400
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, v))
        elif k == "gemini_api_key":
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, str(v or "").strip()))
        else:
            return jsonify({"error": f"Unknown setting: {k}."}), 400
    conn.commit()
    audit("update_settings", ",".join(b.keys()))
    # Return updated settings
    return api_settings()


@app.route("/api/export")
@caregiver_required
def api_export():
    audit("export_all")
    people = [person_public(r, True) for r in db().execute("SELECT * FROM people").fetchall()]
    memories = [mem_public(r, True, True) for r in db().execute("SELECT * FROM memories").fetchall()]
    return jsonify({
        "example_data": False,
        "people": people,
        "relations": [dict(r) for r in db().execute("SELECT * FROM relations").fetchall()],
        "places": [dict(r) for r in db().execute("SELECT * FROM places").fetchall()],
        "memories": memories,
        "reminders": [dict(r) for r in db().execute("SELECT * FROM reminders").fetchall()],
        "alerts": [dict(r) for r in db().execute("SELECT * FROM alerts").fetchall()],
        "care_notes": [dict(r) for r in db().execute("SELECT * FROM notes").fetchall()],
        "audit_log": [dict(r) for r in db().execute("SELECT * FROM audit").fetchall()],
    })


@app.route("/api/erase-all", methods=["POST"])
@caregiver_required
def api_erase_all():
    b = request.get_json(force=True, silent=True) or {}
    if b.get("confirm") != "ERASE":
        return jsonify({"error": 'Send {"confirm": "ERASE"} to permanently delete all data.'}), 400
    conn = db()
    for t in ("faces", "memory_people", "memories", "places", "relations",
              "reminders", "reminder_log", "alerts", "interactions", "notes",
              "schedules", "voice_messages", "voices", "people"):
        conn.execute(f"DELETE FROM {t}")
    conn.execute("DELETE FROM settings WHERE key NOT IN ('pin', 'patient_name')")
    conn.commit()
    audit("erase_all")
    return jsonify({"erased": True})


# ──────────────────────────────────────────────
# /api/speak  — Text-to-Speech (cloned voice)
# ──────────────────────────────────────────────
@app.route("/api/speak", methods=["POST"])
def api_speak():
    """
    Synthesize speech. If person_id is given and that person has a
    cloned voice profile, speak in their voice. Otherwise use EchoBond's
    default voice. Returns audio/mpeg bytes.
    """
    b = request.get_json(force=True, silent=True) or {}
    text = (b.get("text") or "").strip()[:500]
    if not text:
        return jsonify({"error": "No text to speak."}), 400

    person_id = b.get("person_id")
    lang = b.get("lang", setting("voice_lang") or "en-IN")

    try:
        if person_id:
            p = person_row(int(person_id))
            if p and p["consent_voice_clone"] and p["has_voice"] and p["voice_pitch"]:
                audio, profile = speak_as_person(text, dict(p), lang)
                return Response(
                    audio,
                    mimetype="audio/mpeg",
                    headers={
                        "X-Voice-Cloned": "true",
                        "X-Voice-Label": profile["label"],
                        "X-Voice-Person": str(person_id),
                    },
                )

        # Default: EchoBond system voice
        audio = speak_as_echobond(text)
        return Response(
            audio,
            mimetype="audio/mpeg",
            headers={"X-Voice-Cloned": "false", "X-Voice-Label": "EchoBond"},
        )
    except Exception as e:
        # If TTS fails (no internet for edge-tts, etc.), return 204
        # so the frontend falls back to browser speechSynthesis
        return "", 204


# ──────────────────────────────────────────────
# /api/people/<pid>/voice  — Voice Sample Management
# ──────────────────────────────────────────────
@app.route("/api/people/<int:pid>/voice", methods=["GET"])
@caregiver_required
def api_get_voice(pid):
    """Get a person's voice profile and sample info."""
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404

    voice_row = db().execute(
        "SELECT * FROM voices WHERE person_id = ? ORDER BY id DESC LIMIT 1", (pid,)
    ).fetchone()

    lang = setting("voice_lang") or "en-IN"
    profile = build_voice_profile(dict(p), lang)

    return jsonify({
        "person_id": pid,
        "name": p["name"],
        "has_voice": bool(p["has_voice"]),
        "consent_voice_clone": bool(p["consent_voice_clone"]),
        "voice_pitch": p["voice_pitch"],
        "sample_url": voice_row["audio_url"] if voice_row else None,
        "sample_date": voice_row["created_at"] if voice_row else None,
        "profile": profile,
        "voice_messages": [
            dict(vm) for vm in db().execute(
                "SELECT * FROM voice_messages WHERE person_id = ? AND active = 1", (pid,)
            ).fetchall()
        ],
    })


@app.route("/api/people/<int:pid>/voice", methods=["POST"])
@caregiver_required
def api_upload_voice(pid):
    """
    Upload or record a voice sample for a person.
    Analyzes pitch, saves the file, updates the voice profile.
    """
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404
    if not p["consent_voice_clone"]:
        return jsonify({"error": f"{p['name']} hasn't agreed to voice cloning. Enable voice consent first."}), 422

    audio_file = request.files.get("voice") or request.files.get("file") or request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file was attached."}), 400

    audio_bytes = audio_file.read()
    if len(audio_bytes) < 1000:
        return jsonify({"error": "The audio file is too small. Please record at least 5 seconds of speech."}), 400

    # Try to convert to WAV for analysis
    wav_bytes = convert_to_wav(audio_bytes, audio_file.filename or "")

    # Analyze the voice sample
    analysis = analyze_audio_bytes(wav_bytes)

    if not analysis["ok"]:
        return jsonify({"error": analysis["problem"], "analysis": analysis}), 400

    # Save the audio file
    ext = Path(audio_file.filename or "voice.wav").suffix or ".wav"
    fname = f"voice_person_{pid}{ext}"
    fpath = UPLOADS_DIR / fname
    with open(fpath, "wb") as wf:
        wf.write(audio_bytes)
    audio_url = f"/uploads/{fname}"

    conn = db()

    # Remove old voice samples for this person
    conn.execute("DELETE FROM voices WHERE person_id = ?", (pid,))

    # Insert new voice record
    conn.execute(
        "INSERT INTO voices (person_id, audio_url, created_at) VALUES (?, ?, ?)",
        (pid, audio_url, now_iso())
    )

    # Update person's voice info
    conn.execute(
        "UPDATE people SET has_voice = 1, voice_pitch = ? WHERE id = ?",
        (analysis["pitch_hz"], pid)
    )
    conn.commit()

    audit("upload_voice", f"person:{pid}")

    lang = setting("voice_lang") or "en-IN"
    profile = build_voice_profile(
        {**dict(person_row(pid)), "voice_pitch": analysis["pitch_hz"], "has_voice": True, "consent_voice_clone": True},
        lang,
    )

    return jsonify({
        "stored": True,
        "analysis": analysis,
        "profile": profile,
        "audio_url": audio_url,
    }), 201


@app.route("/api/people/<int:pid>/voice", methods=["DELETE"])
@caregiver_required
def api_delete_voice(pid):
    """Remove a person's voice sample and profile."""
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404

    conn = db()
    conn.execute("DELETE FROM voices WHERE person_id = ?", (pid,))
    conn.execute("UPDATE people SET has_voice = 0, voice_pitch = NULL WHERE id = ?", (pid,))
    conn.commit()
    audit("delete_voice", f"person:{pid}")
    return jsonify({"deleted": True})


@app.route("/api/people/<int:pid>/voice/test", methods=["POST"])
@caregiver_required
def api_test_voice(pid):
    """
    Test the cloned voice: synthesize a sample sentence and return audio.
    """
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404
    if not p["has_voice"] or not p["voice_pitch"]:
        return jsonify({"error": "No voice sample uploaded yet."}), 422

    b = request.get_json(force=True, silent=True) or {}
    text = (b.get("text") or "").strip()[:300]
    if not text:
        name = display_name(p["name"])
        pn = patient_name()
        text = f"Namaste {pn}, mai {name} bol raha hu. Aap apna khayal rakhna, mai jaldi aaunga."

    lang = b.get("lang", setting("voice_lang") or "en-IN")

    try:
        audio, profile = speak_as_person(text, dict(p), lang)
        return Response(
            audio,
            mimetype="audio/mpeg",
            headers={
                "X-Voice-Cloned": "true",
                "X-Voice-Label": profile["label"],
            },
        )
    except Exception as e:
        return jsonify({"error": f"Voice synthesis failed: {str(e)}"}), 500


# ──────────────────────────────────────────────
# /api/people/<pid>/voice-messages  — Approved Messages
# ──────────────────────────────────────────────
@app.route("/api/people/<int:pid>/voice-messages", methods=["POST"])
@caregiver_required
def api_add_voice_message(pid):
    """Add an approved voice message template for a person."""
    p = person_row(pid)
    if not p:
        return jsonify({"error": "Person not found."}), 404
    if not p["consent_voice_clone"] or p["status"] != "active":
        return jsonify({"error": f"{p['name']} hasn't agreed to voice cloning."}), 422

    b = request.get_json(force=True, silent=True) or {}
    template = (b.get("template") or "").strip()
    if not template:
        return jsonify({"error": "Message text is required."}), 400

    approved_by = (b.get("approved_by") or "").strip()
    if not approved_by:
        return jsonify({"error": "Record who approved these words."}), 400

    kind = b.get("kind", "away")
    if kind not in ("away", "home", "reassure"):
        kind = "away"

    cur = db().execute(
        "INSERT INTO voice_messages (person_id, kind, template, approved_by, approved_at, active) VALUES (?, ?, ?, ?, ?, 1)",
        (pid, kind, template, approved_by, now_iso())
    )
    db().commit()
    mid = cur.lastrowid
    audit("approve_voice_message", f"person:{pid}:message:{mid}")
    return jsonify(dict(db().execute("SELECT * FROM voice_messages WHERE id = ?", (mid,)).fetchone())), 201


@app.route("/api/voice-messages/<int:mid>", methods=["DELETE"])
@caregiver_required
def api_delete_voice_message(mid):
    """Deactivate a voice message."""
    db().execute("UPDATE voice_messages SET active = 0 WHERE id = ?", (mid,))
    db().commit()
    audit("remove_voice_message", f"message:{mid}")
    return jsonify({"deleted": mid})


# ──────────────────────────────────────────────
# /api/voice-chat  — Familiar Voice Companion
# ──────────────────────────────────────────────
@app.route("/api/voice-chat", methods=["POST"])
def api_voice_chat():
    """
    Voice companion: when the patient asks about or wants to talk to
    a family member, respond using their cloned voice if available.
    Falls back to normal chat for safety/unknown/no-target queries.
    """
    carer = is_caregiver()
    b = request.get_json(force=True, silent=True) or {}
    msg = (b.get("message") or "").strip()[:500]
    if not msg:
        return jsonify({"error": "Ask a question, for example 'Where is Rohan?'"}), 400

    preview = bool(b.get("preview")) and carer
    session_key = b.get("session", "")
    people = load_all_people()
    active = [p for p in people if p["status"] == "active"]
    reminders = load_all_reminders()
    schedules = load_all_schedules()

    # Find target person
    target = None
    if b.get("person_id"):
        target = next((p for p in active if p["id"] == int(b["person_id"])), None)

    if not target:
        low = msg.lower()
        for p in active:
            names_to_check = [p["name"].split()[0].lower()]
            if p.get("nickname"):
                names_to_check.append(p["nickname"].lower().replace(".", ""))
            for n in names_to_check:
                if n and re.search(r"\b" + re.escape(n) + r"\b", low):
                    target = p
                    break
            if target:
                break

    # Safety check or no clear target -> fall back to normal chat
    safety_cat = detect_support_category(msg)
    is_talk = bool(TALK_PATTERN.search(msg))
    is_where = bool(WHERE_PATTERN.search(msg))

    if safety_cat or not target or (not is_talk and not is_where and not b.get("person_id")):
        # Delegate to normal chat engine
        memories = load_all_memories()
        places = load_all_places()
        ctx = sessions.get(session_key, {"person_ids": [], "memory_ids": [], "told_ids": []})
        engine = ConversationEngine(db(), patient_name())
        result, alert_data, new_ctx = engine.answer_query(msg, ctx, people, memories, places, reminders, schedules)
        if new_ctx:
            sessions[session_key] = new_ctx
        if alert_data and not preview:
            add_alert(alert_data["level"], alert_data["kind"], alert_data["reason"], alert_data["detail"], alert_data.get("ref", ""))

        segments = [make_segment("echobond", result["text"])]
        return jsonify({
            "segments": segments,
            "text": result["text"],
            "person": None,
            "mode": "echobond",
            "why": result.get("why", []),
            "alert": result.get("alert"),
            "category": result.get("category"),
            "engine": result.get("engine"),
            **({"preview": True} if preview else {}),
        })

    first = display_name(target["name"])
    sch = get_person_schedule_now(target["id"], schedules)
    memories = load_all_memories()

    # Dynamic, natural first-person conversational response from target
    person_text, engine_used = generate_family_voice_reply(
        msg, target, patient_name(), sch, memories, db()
    )

    has_cloned_voice = bool(target.get("has_voice") and target.get("consent_voice_clone") and target.get("voice_pitch"))
    mode = "cloned" if has_cloned_voice else "fallback"

    segments = [make_segment("person", person_text, target, has_cloned_voice)]
    why = [{
        "type": "persona",
        "text": f"Spoken directly in {first}'s personal voice via {'Gemini AI' if engine_used == 'gemini' else 'Adaptive Family Persona Engine'}"
    }]

    # If patient specifically wants to initiate a real phone call
    if is_talk and not b.get("person_id"):
        carer_name = get_carer_display()
        if carer_name != first:
            talk_text = f"I've let {carer_name} know you'd like to talk to {first}."
        else:
            talk_text = f"I've let {first} know you'd like to talk."
        segments.append(make_segment("echobond", talk_text))

        if not preview:
            add_alert("info", "wants_to_talk",
                      f"{patient_name()} would like to talk to {first}",
                      msg, str(target["id"]))

        why.append({
            "type": "rule",
            "text": "EchoBond lets the caregiver know when the patient asks for a real phone call",
        })

    full_text = person_text

    if not preview:
        norm = normalize_question(msg)
        db().execute(
            "INSERT INTO interactions (kind, person_id, outcome, question, normalized, category, memory_ids, session, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("chat", target["id"], "answered", msg, norm, "",
             "[]", session_key, f"Q: {msg} | A ({mode}): {full_text[:120]}", now_iso())
        )
        db().commit()

    return jsonify({
        "segments": segments,
        "text": full_text,
        "person": {
            "id": target["id"],
            "name": target["name"],
            "relationship": target["relationship"],
        },
        "mode": mode,
        "why": why,
        "alert": "info" if is_talk else None,
        "category": None,
        "engine": "companion",
        **({"preview": True} if preview else {}),
    })


# ──────────────────────────────────────────────
# /api/voices  — List all people with voice profiles
# ──────────────────────────────────────────────
@app.route("/api/voices")
def api_voices():
    """List all active people who have voice cloning enabled."""
    people = db().execute(
        "SELECT * FROM people WHERE status = 'active' AND consent_voice_clone = 1"
    ).fetchall()

    lang = setting("voice_lang") or "en-IN"
    result = []
    for p in people:
        pd = dict(p)
        profile = build_voice_profile(pd, lang)
        voice_row = db().execute(
            "SELECT audio_url, created_at FROM voices WHERE person_id = ? ORDER BY id DESC LIMIT 1",
            (p["id"],)
        ).fetchone()
        msg_count = db().execute(
            "SELECT COUNT(*) as c FROM voice_messages WHERE person_id = ? AND active = 1",
            (p["id"],)
        ).fetchone()["c"]
        result.append({
            "id": p["id"],
            "name": p["name"],
            "relationship": p["relationship"],
            "photo_url": p["photo"] if p["consent_photo"] else None,
            "avatar_color": p["avatar_color"],
            "has_voice": bool(p["has_voice"]),
            "voice_pitch": p["voice_pitch"],
            "profile": profile,
            "sample_url": voice_row["audio_url"] if voice_row else None,
            "message_count": msg_count,
        })
    return jsonify(result)


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  EchoBond Backend Server")
    print("=" * 50)
    seed_database()
    with app.app_context():
        pin_val = setting("pin")
        pat_val = patient_name()
    print(f"\n  Server running at http://localhost:{PORT}")
    print(f"  Caregiver PIN: {pin_val}")
    print(f"  Patient: {pat_val}")
    print(f"  Database: {DB_PATH}")
    print(f"\n  Open http://localhost:{PORT} in your browser")
    print("=" * 50)
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
