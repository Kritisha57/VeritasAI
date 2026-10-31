"""Data structures for multimodal document ingestion and hierarchical chunking."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedTable:
    """Structured table from PDF or Excel ready for SQLite insertion and SQL querying."""

    table_id: str
    doc_id: str
    sql_table_name: str
    dataframe: Any  # pandas.DataFrame
    schema_json: str
    summary_text: str
    page_number: int = 1


@dataclass
class ParentChunk:
    """Large context block (~1200 tokens) stored unembedded in SQLite for LLM synthesis."""

    parent_id: str
    doc_id: str
    filename: str
    section_path: str
    page_number: int
    content: str


@dataclass
class ChildChunk:
    """Small retrieval unit (~300 tokens) with contextual breadcrumb embedded in FAISS."""

    child_id: str
    parent_id: str
    doc_id: str
    breadcrumb_text: str
    modality: str  # "text" | "table_summary" | "figure_caption"
    page_number: int = 1
    image_path: str | None = None


@dataclass
class IngestionResult:
    """Complete output of parsing a single document across all modalities."""

    doc_id: str
    filename: str
    file_type: str
    file_hash: str
    parent_chunks: list[ParentChunk] = field(default_factory=list)
    child_chunks: list[ChildChunk] = field(default_factory=list)
    extracted_tables: list[ExtractedTable] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the ingestion result (including DataFrame rows) into a JSON-ready dict."""
        from dataclasses import asdict

        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_hash": self.file_hash,
            "counts": {
                "parent_chunks": len(self.parent_chunks),
                "child_chunks": len(self.child_chunks),
                "extracted_tables": len(self.extracted_tables),
            },
            "parent_chunks": [asdict(p) for p in self.parent_chunks],
            "child_chunks": [asdict(c) for c in self.child_chunks],
            "extracted_tables": [
                {
                    "table_id": t.table_id,
                    "doc_id": t.doc_id,
                    "sql_table_name": t.sql_table_name,
                    "page_number": t.page_number,
                    "schema_json": t.schema_json,
                    "summary_text": t.summary_text,
                    "rows": t.dataframe.fillna("").to_dict(orient="records"),
                }
                for t in self.extracted_tables
            ],
        }

