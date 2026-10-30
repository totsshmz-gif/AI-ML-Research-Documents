from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

#*****************
# Project Paths  *
#*****************

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
CHROMA_DIR = BASE_DIR / "chroma_db"

#*****************************
# Application Configuration  *
#*****************************

class Settings(BaseSettings):

    # LLM
    llm_model: str = "qwen2.5:3b"

    # Embeddings
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 150

    top_k: int = 6
    distance_threshold: float = 2.5

    # Chroma
    collection_name: str = "research_papers"

    # Hugging Face
    hf_token: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

#*****************
# Load Settings  *
#*****************

settings = Settings()

#**********************************
# Backward-Compatible Constants   *
#**********************************
'''Existing project files can continue using:
TOP_K, LLM_MODEL, EMBEDDING_MODEL, etc.
without requiring changes everywhere else.'''

COLLECTION_NAME = settings.collection_name

EMBEDDING_MODEL = settings.embedding_model

LLM_MODEL = settings.llm_model

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

TOP_K = settings.top_k

DISTANCE_THRESHOLD = settings.distance_threshold

HF_TOKEN = settings.hf_token