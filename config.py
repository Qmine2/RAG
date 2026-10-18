"""Configuration centrale du projet RAG Telecom."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    # huggingface_hub lit cette variable pour télécharger le modèle d'embeddings
    os.environ["HF_TOKEN"] = HF_TOKEN

# Sources de données
FAQ_PATH = BASE_DIR / "faq.csv"
PDF_PATH = BASE_DIR / "telecom_guide.pdf"
DB_PATH = BASE_DIR / "tickets.db"

# Base vectorielle
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "telecom_kb"

# Modèles
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# Découpage du PDF (en mots)
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40

# Recherche
TOP_K = 5
