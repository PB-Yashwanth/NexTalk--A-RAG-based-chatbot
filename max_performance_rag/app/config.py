"""
Application configuration.
All settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "documents"

# === ENVIRONMENT ===
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

# === EMBEDDING MODEL ===
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2

# === VECTOR STORE (ChromaDB) ===
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")

# === LLM ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# === RAG ===
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.30"))

# === MEMORY (PERSISTENT) ===
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "6"))
MEMORY_DIR = BASE_DIR / "chat_memory"
MEMORY_DB_PATH = MEMORY_DIR / "memory.db"

# === FILE UPLOAD ===
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = [e.strip() for e in os.getenv("ALLOWED_EXTENSIONS", ".txt,.pdf,.docx").split(",") if e.strip()]

# === CORS ===
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:4200,http://localhost:8501,http://127.0.0.1:8501"
).split(",") if o.strip()]

# === Rate limiting ===
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "60/minute")
RATE_LIMIT_INGEST = os.getenv("RATE_LIMIT_INGEST", "10/minute")

# === API Key Auth ===
API_KEY = os.getenv("API_KEY", "").strip()

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()