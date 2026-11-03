"""File format router dispatching PDFs, Excel/CSV, Markdown, and images to specialized parsers."""

import hashlib
import shutil
from pathlib import Path
from typing import Callable
from config.settings import settings
from ingestion.chunker import HierarchicalChunker
from ingestion.markdown_parser import MarkdownParser
from ingestion.models import IngestionResult
from ingestion.pdf_parser import PDFParser
from ingestion.tabular_parser import TabularParser


class IngestionRouter:
    """Computes deterministic document hashes and routes files by extension."""

    TABULAR_EXTS = {".xlsx", ".xls", ".csv"}
    MARKDOWN_EXTS = {".md", ".markdown", ".txt"}
    PDF_EXTS = {".pdf"}
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(
        self,
        chunker: HierarchicalChunker | None = None,
        figure_captioner: Callable[[Path, str], str] | None = None,
        figures_dir: Path = settings.figures_dir,
    ) -> None:
        self.chunker = chunker or HierarchicalChunker()
        self.figures_dir = figures_dir
        self.figure_captioner = figure_captioner
        self.md_parser = MarkdownParser(self.chunker)
        self.tabular_parser = TabularParser(self.chunker)
        self.pdf_parser = PDFParser(self.chunker, self.tabular_parser, self.figure_captioner, figures_dir)

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """Parses any supported document into ParentChunks, ChildChunks, and ExtractedTables."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = self.compute_sha256(path)
        doc_id = f"doc_{file_hash[:16]}"
        suffix = path.suffix.lower()

        if suffix in self.PDF_EXTS:
            parents, children, tables = self.pdf_parser.parse(path, doc_id)
            file_type = "pdf"
        elif suffix in self.TABULAR_EXTS:
            parents, children, tables = self.tabular_parser.parse(path, doc_id)
            file_type = "tabular"
        elif suffix in self.MARKDOWN_EXTS:
            parents, children = self.md_parser.parse(path, doc_id)
            tables = []
            file_type = "markdown"
        elif suffix in self.IMAGE_EXTS:
            parents, children = self._parse_standalone_image(path, doc_id)
            tables = []
            file_type = "image"
        else:
            raise ValueError(f"Unsupported file extension: '{suffix}'")

        return IngestionResult(
            doc_id=doc_id,
            filename=path.name,
            file_type=file_type,
            file_hash=file_hash,
            parent_chunks=parents,
            child_chunks=children,
            extracted_tables=tables,
        )

    def _parse_standalone_image(self, path: Path, doc_id: str):
        """Copies a standalone chart/image to figures directory and indexes its rich VLM description."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        if path.parent.resolve() == self.figures_dir.resolve():
            dest = path
        else:
            dest = self.figures_dir / f"{doc_id[:8]}_{path.name}"
            if path != dest:
                shutil.copy2(path, dest)

        desc = (
            self.figure_captioner(dest, path.stem)
            if self.figure_captioner
            else f"Standalone chart or image file '{path.name}'."
        )
        parents, children = self.chunker.chunk_section(
            doc_id=doc_id,
            filename=path.name,
            section_path="Standalone Figure",
            page_number=1,
            text=f"[Image: {dest.name}] {desc}\n![{path.name}](/figures/{dest.name})",
            modality="figure_caption",
        )
        for child in children:
            child.image_path = str(dest)
        return parents, children

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 hex digest of a file in 64KB chunks."""
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
