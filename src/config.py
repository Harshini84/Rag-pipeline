from pathlib import Path
import os

try:
    # optional: use python-dotenv if available
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]

# Documents directory (default: project root / data)
DOCS_PATH = os.getenv("DOCS_PATH", str(BASE_DIR / "data"))

# Retrieval / ranking defaults
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.1"))

# If set, use only these files (comma-separated list). Otherwise None
_spec = os.getenv("SPECIFIC_FILES", "")
if _spec:
    SPECIFIC_FILES = [s.strip() for s in _spec.split(",") if s.strip()]
else:
    SPECIFIC_FILES = None

# Databricks / model serving credentials used by `document_loader.py` and `llm.py`
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_MODEL_ENDPOINT = os.getenv("DATABRICKS_MODEL_ENDPOINT", "")

__all__ = [
    "BASE_DIR",
    "DOCS_PATH",
    "TOP_K",
    "MIN_SCORE_THRESHOLD",
    "SPECIFIC_FILES",
    "DATABRICKS_TOKEN",
    "DATABRICKS_HOST",
    "DATABRICKS_MODEL_ENDPOINT",
]
