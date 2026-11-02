import sqlite3
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from .config import DB_PATH, DEFAULT_PIN, PATIENT_NAME

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT,
        name TEXT NOT NULL,
        relationship TEXT NOT NULL,
        nickname TEXT DEFAULT '',
        description TEXT DEFAULT '',
        status TEXT DEFAULT 'active',
        consent_photo INTEGER DEFAULT 0,
        consent_voice_clone INTEGER DEFAULT 0,
        consent_given_by TEXT DEFAULT '',
        consent_date TEXT DEFAULT '',
        avatar_color TEXT DEFAULT '#8fa3b8',
        photo TEXT,
        has_voice INTEGER DEFAULT 0,
        birthday TEXT DEFAULT '',
        favourites TEXT DEFAULT '',
        notes TEXT DEFAULT '',
        occupation TEXT DEFAULT '',
        hobbies TEXT DEFAULT '',
        connection TEXT DEFAULT '',
        topics TEXT DEFAULT '',
        calls_patient TEXT DEFAULT '',
        voice_pitch REAL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        embedding TEXT NOT NULL,
        photo_url TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS places (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        region TEXT DEFAULT '',
        lat REAL,
        lng REAL,
        note TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        memory_date TEXT DEFAULT '',
        year INTEGER,
        life_stage TEXT DEFAULT 'recent',
        category TEXT DEFAULT 'family',
        place_id INTEGER,
        privacy TEXT DEFAULT 'normal',
        song_title TEXT DEFAULT '',
        song_artist TEXT DEFAULT '',
        song_url TEXT,
        photo_url TEXT,
        source TEXT DEFAULT 'caregiver',
        created_at TEXT NOT NULL,
        FOREIGN KEY (place_id) REFERENCES places(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS memory_people (
        memory_id INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        PRIMARY KEY (memory_id, person_id),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        related_id INTEGER NOT NULL,
        kind TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE,
        FOREIGN KEY (related_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        time_of_day TEXT NOT NULL,
        kind TEXT DEFAULT 'routine',
        days TEXT DEFAULT 'daily',
        active INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS reminder_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reminder_id INTEGER NOT NULL,
        status TEXT DEFAULT 'done',
        created_at TEXT NOT NULL,
        FOREIGN KEY (reminder_id) REFERENCES reminders(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT DEFAULT 'info',
        kind TEXT DEFAULT '',
        reason TEXT NOT NULL,
        detail TEXT NOT NULL,
        acknowledged INTEGER DEFAULT 0,
        ref TEXT DEFAULT '',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        person_id INTEGER,
        outcome TEXT NOT NULL,
        score REAL,
        question TEXT,
        normalized TEXT,
        category TEXT,
        memory_ids TEXT,
        session TEXT,
        summary TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mood TEXT NOT NULL,
        note TEXT DEFAULT '',
        author TEXT DEFAULT 'caregiver',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor TEXT DEFAULT 'caregiver',
        action TEXT NOT NULL,
        target TEXT DEFAULT '',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        days TEXT DEFAULT 'daily',
        start_at TEXT NOT NULL,
        end_at TEXT NOT NULL,
        place TEXT NOT NULL,
        back_at TEXT DEFAULT '',
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS voice_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        kind TEXT DEFAULT 'away',
        template TEXT NOT NULL,
        approved_by TEXT NOT NULL,
        approved_at TEXT NOT NULL,
        active INTEGER DEFAULT 1,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS voices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        audio_url TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );
    """)

    # Default settings
    default_settings = {
        "pin": DEFAULT_PIN,
        "patient_name": PATIENT_NAME,
        "face_recognize_threshold": "0.90",
        "voice_lang": "en-IN",
        "voice_rate": "0.86",
        "repeat_window_hours": "24",
        "repeat_threshold": "3",
        "default_privacy": "normal"
    }

    for k, v in default_settings.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
        
    conn.commit()
    conn.close()

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
