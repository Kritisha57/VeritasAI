"""Lightweight figure caption helper (uses extracted PDF captions instead of heavy VLM calls)."""

import base64
from pathlib import Path


def encode_image_base64(image_path: Path) -> str:
    """Encodes an image file on disk into a base64 data URI string."""
    ext = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def describe_image(image_path: Path, caption_hint: str = "") -> str:
    """Returns the native figure caption hint or a simple filename fallback."""
    return caption_hint or f"Figure from {image_path.name}"
