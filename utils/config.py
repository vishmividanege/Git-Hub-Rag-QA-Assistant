from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstores"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Chunking ----------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------- Retrieval ----------
TOP_K = 8
SEARCH_TYPE = "cosine similarity"

# ---------- Models ----------

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.2
MAX_CHAT_HISTORY = 10

# ---------- Guardrails ----------

ENABLE_SECRET_SCANNING = True

SKIP_FILES_WITH_SECRETS = True

EXCLUDED_FILENAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "secrets.json", "credentials.json", "credentials.yml", "credentials.yaml",
    "config.secret.json", "serviceAccountKey.json",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
}

EXCLUDED_SECRET_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".cer", ".cert", ".crt"}

# Rate limiting for embeddings 
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_BATCH_DELAY = 0

# Directories to ignore
EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", "dist", "build", "venv", 
    ".venv", "env", ".env_vars", "target", "bin", "obj", ".idea", ".vscode"
}

# Extensions to ignore 
EXCLUDED_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    # Media
    ".mp3", ".mp4", ".mov", ".avi", ".wav",
    # Archives
    ".zip", ".tar", ".gz", ".7z", ".rar",
    # Binaries/Compiled
    ".exe", ".dll", ".so", ".bin", ".pyc", ".pyd", ".o", ".a", ".lib",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Data/Large
    ".csv", ".sqlite", ".db", ".parquet", ".pickle",
    
}
