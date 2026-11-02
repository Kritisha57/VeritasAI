"""VLM image captioner using Qwen/Qwen3.5-27B via Hugging Face Inference API."""

import base64
import os
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_VLM_MODEL = "Qwen/Qwen3.5-27B"


def encode_image_base64(image_path: Path) -> str:
    """Encodes an image file on disk into a base64 data URI string."""
    ext = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def describe_image(image_path: Path, caption_hint: str = "") -> str:
    """Generates a rich, searchable description of a figure using Qwen/Qwen3.5-27B."""
    hf_key = os.getenv("HUGGINGFACE_KEY") or os.getenv("HF_TOKEN", "")
    model = os.getenv("VLM_MODEL", DEFAULT_VLM_MODEL)
    if not hf_key or not image_path.exists():
        return caption_hint or f"Figure from {image_path.name}"

    data_uri = encode_image_base64(image_path)
    prompt = (
        "Provide a comprehensive, rich technical description of this research figure, chart, or architecture diagram. "
        "Include all visible labels, components, arrows, data flows, axes, trends, and numerical values so it can be "
        "accurately retrieved by a text search engine."
    )
    if caption_hint:
        prompt += f"\nOriginal figure caption: {caption_hint}"

    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": 500,
        "temperature": 0.2,
    }
    try:
        resp = httpx.post(HF_API_URL, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f" -> [VLM Warning] Could not caption {image_path.name} with {model}: {e}")
        return caption_hint or f"Figure image {image_path.name}"


def enrich_existing_figures(
    figures_dir: Path,
    sqlite_store,
    faiss_store,
    embedder,
) -> int:
    """Passes all images in uploads/figures through Qwen/Qwen3.5-27B and stores rich descriptions in SQLite/FAISS."""
    if not figures_dir.exists():
        return 0

    image_files = sorted(
        f for f in figures_dir.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not image_files:
        return 0

    updated = 0
    new_child_ids: list[str] = []
    new_child_texts: list[str] = []

    with sqlite_store._get_connection() as conn:
        for img_path in image_files:
            rows = conn.execute(
                "SELECT parent_id, doc_id, filename, section_path, page_number, content "
                "FROM parent_chunks WHERE content LIKE ?",
                (f"%{img_path.name}%",),
            ).fetchall()

            # Skip if already enriched with a long VLM description (>350 chars)
            if rows and all(len(r["content"]) > 350 for r in rows):
                continue

            print(f" -> Generating rich VLM description for figure: {img_path.name}...")
            rich_desc = describe_image(img_path)
            if not rich_desc:
                continue

            for r in rows:
                new_content = (
                    f"[Figure | Image: {img_path.name}] {rich_desc}\n"
                    f"![{r['section_path']}](/figures/{img_path.name})"
                )
                conn.execute(
                    "UPDATE parent_chunks SET content = ? WHERE parent_id = ?",
                    (new_content, r["parent_id"]),
                )
                c_rows = conn.execute(
                    "SELECT child_id FROM child_chunks WHERE parent_id = ?",
                    (r["parent_id"],),
                ).fetchall()
                for cr in c_rows:
                    b_text = f"[Doc: {r['filename']} | Section: {r['section_path']} | Page: {r['page_number']}] {new_content}"
                    conn.execute(
                        "UPDATE child_chunks SET breadcrumb_text = ? WHERE child_id = ?",
                        (b_text, cr["child_id"]),
                    )
                    new_child_ids.append(cr["child_id"])
                    new_child_texts.append(b_text)
                updated += 1
        conn.commit()

    if new_child_texts and embedder and faiss_store:
        vecs = embedder.embed_documents(new_child_texts)
        faiss_store.add(new_child_ids, vecs)

    return updated
