"""Centralized application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable configuration for VeritasAI modules."""

    # Paths
    data_dir: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    upload_dir: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
    figures_dir: Path = BASE_DIR / os.getenv("FIGURES_DIR", "data/uploads/figures")
    index_dir: Path = BASE_DIR / os.getenv("INDEX_DIR", "data/indexes")
    db_path: Path = BASE_DIR / os.getenv("DB_PATH", "data/db/veritas.sqlite")

    # Ingestion
    parent_chunk_tokens: int = int(os.getenv("PARENT_CHUNK_TOKENS", "1200"))
    child_chunk_tokens: int = int(os.getenv("CHILD_CHUNK_TOKENS", "300"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
    vlm_model: str = os.getenv("VLM_MODEL", "ibm-granite/granite-docling-258M")
    docling_image_scale: float = float(os.getenv("DOCLING_IMAGE_SCALE", "2.0"))
    docling_do_ocr: bool = os.getenv("DOCLING_DO_OCR", "false").lower() == "true"

    # Embedding & Reranking (HLD: Qwen/Qwen3-Embedding-8B, Qwen/Qwen3-Reranker-4B)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")

    # Generation LLM via Groq (HLD: openai/gpt-oss-120b)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


settings = Settings()
