from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstores"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Chunking ----------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---------- Retrieval ----------
TOP_K = 4
SEARCH_TYPE = "similarity"

# ---------- Models ----------
# Using Gemini models as per previous requirement for a working Gemini version
EMBEDDING_MODEL_NAME = "models/gemini-embedding-2-preview"
LLM_MODEL_NAME = "models/gemini-flash-latest"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2

# File types to process
SUPPORTED_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", 
    ".md", ".txt", ".json", ".yaml", ".yml",
    ".java", ".cpp", ".c", ".h", ".go", ".rs",
    ".php", ".rb", ".sh", ".sql"
)
