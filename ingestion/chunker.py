"""Hierarchical Parent-Child chunker with contextual breadcrumb injection."""

import hashlib
import re
from config.settings import settings
from ingestion.models import ChildChunk, ParentChunk


class HierarchicalChunker:
    """Splits document sections into Parent Chunks (context) and Child Chunks (retrieval)."""

    def __init__(
        self,
        parent_tokens: int = settings.parent_chunk_tokens,
        child_tokens: int = settings.child_chunk_tokens,
        overlap_tokens: int = settings.chunk_overlap_tokens,
    ) -> None:
        self.parent_words = max(1, int(parent_tokens * 0.75))
        self.child_words = max(1, int(child_tokens * 0.75))
        self.overlap_words = max(0, int(overlap_tokens * 0.75))

    @staticmethod
    def format_breadcrumb(filename: str, section_path: str, page: int, text: str) -> str:
        """Prepends document, section hierarchy, and page metadata to a child chunk."""
        section = section_path.strip() or "Root"
        return f"[Doc: {filename} | Section: {section} | Page: {page}] {text.strip()}"

    @staticmethod
    def _make_id(prefix: str, doc_id: str, index: int, text: str) -> str:
        digest = hashlib.sha1(f"{doc_id}:{index}:{text[:120]}".encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{doc_id[:8]}_{index}_{digest}"

    def chunk_section(
        self,
        doc_id: str,
        filename: str,
        section_path: str,
        page_number: int,
        text: str,
        modality: str = "text",
        start_index: int = 0,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Splits a section into parent chunks and overlapping breadcrumb-prefixed child chunks."""
        cleaned = text.strip()
        if not cleaned:
            return [], []

        parent_texts = self._split_by_words(cleaned, self.parent_words, overlap=0)
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for p_offset, p_text in enumerate(parent_texts):
            p_idx = start_index + p_offset
            parent_id = self._make_id("par", doc_id, p_idx, p_text)
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    filename=filename,
                    section_path=section_path,
                    page_number=page_number,
                    content=p_text,
                )
            )

            child_texts = self._split_by_words(p_text, self.child_words, self.overlap_words)
            for c_offset, c_text in enumerate(child_texts):
                child_id = self._make_id("chd", parent_id, c_offset, c_text)
                breadcrumb_text = self.format_breadcrumb(filename, section_path, page_number, c_text)
                children.append(
                    ChildChunk(
                        child_id=child_id,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        breadcrumb_text=breadcrumb_text,
                        modality=modality,
                        page_number=page_number,
                    )
                )

        return parents, children

    @staticmethod
    def _split_by_words(text: str, max_words: int, overlap: int) -> list[str]:
        """Splits text along paragraph/sentence boundaries while respecting word limits."""
        words = text.split()
        if len(words) <= max_words:
            return [text.strip()]

        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        chunks: list[str] = []
        current_words: list[str] = []

        for block in blocks:
            block_words = block.split()
            if len(current_words) + len(block_words) <= max_words:
                current_words.extend(block_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = current_words[-overlap:] if overlap > 0 else []
                while len(block_words) > max_words:
                    take = max_words - len(current_words)
                    current_words.extend(block_words[:take])
                    chunks.append(" ".join(current_words))
                    block_words = block_words[take:]
                    current_words = current_words[-overlap:] if overlap > 0 else []
                current_words.extend(block_words)

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks
