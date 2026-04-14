import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]

# Load the repo-level .env so the backend works when launched from either the
# repo root or inside the DataLens folder.
load_dotenv(REPO_ROOT / ".env")

TEMP_ROOT = ROOT_DIR / "temp"
FRONTEND_ROOT = ROOT_DIR.parent / "frontend"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
OPEN_ROUTER_MODEL = os.getenv("OPEN_ROUTER_MODEL", "z-ai/glm-4.5-air:free")

PDF_EXTENSIONS = {".pdf"}
CSV_EXTENSIONS = {".csv"}

# Default chunking settings from MultiModal_RAG_(No_Framework).ipynb
TEXT_CHUNK_SIZE = 2048
TEXT_CHUNK_OVERLAP = 50
