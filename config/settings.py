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

    data_dir: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    upload_dir: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
    figures_dir: Path = BASE_DIR / os.getenv("FIGURES_DIR", "data/uploads/figures")
    index_dir: Path = BASE_DIR / os.getenv("INDEX_DIR", "data/indexes")
    db_path: Path = BASE_DIR / os.getenv("DB_PATH", "data/db/veritas.sqlite")

    parent_chunk_tokens: int = int(os.getenv("PARENT_CHUNK_TOKENS", "1200"))
    child_chunk_tokens: int = int(os.getenv("CHILD_CHUNK_TOKENS", "300"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
    vlm_model: str = os.getenv("VLM_MODEL", "ibm-granite/granite-docling-258M")
    docling_image_scale: float = float(os.getenv("DOCLING_IMAGE_SCALE", "2.0"))
    docling_do_ocr: bool = os.getenv("DOCLING_DO_OCR", "false").lower() == "true"


settings = Settings()
