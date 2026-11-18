from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", DATA_DIR / "documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", DATA_DIR / "research_assistant.db"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "5"))

def ensure_directories() -> None:
    for path in (DOCUMENTS_DIR, CHROMA_DIR, DATABASE_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
