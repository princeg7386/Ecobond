# EchoBond Backend Configuration
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
ASSETS_DIR = BASE_DIR / "assets"

DATA_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

DB_PATH = DATA_DIR / "echobond.db"
DEFAULT_PIN = os.getenv("ECHOBOND_PIN", "2468")
PATIENT_NAME = os.getenv("ECHOBOND_PATIENT", "Kamala")
PORT = int(os.getenv("PORT", 8080))
HOST = os.getenv("HOST", "0.0.0.0")
