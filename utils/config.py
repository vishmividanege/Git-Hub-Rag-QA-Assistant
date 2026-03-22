from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstores"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Chunking ----------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 120

# ---------- Retrieval ----------
TOP_K = 6
SEARCH_TYPE = "cosine similarity"

# ---------- Models ----------

EMBEDDING_MODEL_NAME = "models/gemini-embedding-2-preview"
LLM_MODEL_NAME = "models/gemini-flash-latest"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2

# Rate limiting for embeddings (Gemini Free Tier: 100 RPM)
EMBEDDING_BATCH_SIZE = 30
EMBEDDING_BATCH_DELAY = 5

# File types to process
SUPPORTED_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", 
    ".md", ".txt", ".json", ".yaml", ".yml",
    ".java", ".cpp", ".c", ".h", ".go", ".rs",
    ".php", ".rb", ".sh", ".sql"
)
