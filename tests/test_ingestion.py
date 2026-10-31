"""Unit and integration tests for Module 1: Multimodal Ingestion."""

import json
from pathlib import Path
import pandas as pd
import pytest
from ingestion.chunker import HierarchicalChunker
from ingestion.markdown_parser import MarkdownParser
from ingestion.router import IngestionRouter
from ingestion.tabular_parser import TabularParser


def test_hierarchical_chunker_creates_parents_and_breadcrumb_children():
    """Verifies Parent-Child splitting and contextual breadcrumb formatting."""
    chunker = HierarchicalChunker(parent_tokens=80, child_tokens=30, overlap_tokens=8)
    long_text = "\n\n".join(
        [f"Paragraph {i} discusses ablation metrics and latency optimization in detail." for i in range(15)]
    )

    parents, children = chunker.chunk_section(
        doc_id="doc_1234567890",
        filename="paper.pdf",
        section_path="3. Experiments > 3.2 Ablation",
        page_number=7,
        text=long_text,
    )

    assert len(parents) >= 2
    assert len(children) > len(parents)
    parent_ids = {p.parent_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids
        assert child.breadcrumb_text.startswith("[Doc: paper.pdf | Section: 3. Experiments > 3.2 Ablation | Page: 7]")


def test_markdown_parser_preserves_fenced_code_and_breadcrumbs(tmp_path: Path):
    """Ensures # comments inside fenced code blocks do not trigger false section splits."""
    md_file = tmp_path / "architecture.md"
    md_file.write_text(
        "# System Overview\n"
        "High level intro.\n\n"
        "## Pipeline Stage\n"
        "```python\n"
        "# This is a python comment, NOT a markdown H1 header!\n"
        "def run():\n"
        "    return 42\n"
        "```\n"
        "Final paragraph under Pipeline Stage.\n",
        encoding="utf-8",
    )

    parser = MarkdownParser()
    parents, children = parser.parse(md_file, doc_id="doc_mdtest01")

    assert len(parents) == 2
    assert parents[0].section_path == "System Overview"
    assert parents[1].section_path == "System Overview > Pipeline Stage"
    assert "# This is a python comment" in parents[1].content
    assert "Final paragraph under Pipeline Stage." in parents[1].content
    assert len(children) >= 2


def test_tabular_parser_csv_and_excel_dual_indexing(tmp_path: Path):
    """Verifies that CSV and multi-sheet Excel files produce both SQL DataFrames and summary chunks."""
    df_q1 = pd.DataFrame(
        {
            "Region Name": ["North America", "Europe", "APAC"],
            "Revenue ($M)": [120.5, 95.0, 140.2],
            "Growth %": [12.1, 8.4, 19.5],
        }
    )
    csv_path = tmp_path / "q1_metrics.csv"
    df_q1.to_csv(csv_path, index=False)

    xlsx_path = tmp_path / "annual_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_q1.to_excel(writer, sheet_name="Q1 Summary", index=False)
        df_q1.to_excel(writer, sheet_name="Q2 Forecast", index=False)

    router = IngestionRouter()

    csv_res = router.ingest_file(csv_path)
    assert csv_res.file_type == "tabular"
    assert len(csv_res.extracted_tables) == 1
    tbl = csv_res.extracted_tables[0]
    assert list(tbl.dataframe.columns) == ["region_name", "revenue_m", "growth"]
    schema = json.loads(tbl.schema_json)
    assert "revenue_m" in schema
    assert csv_res.child_chunks[0].modality == "table_summary"
    assert "revenue_m" in csv_res.child_chunks[0].breadcrumb_text

    xlsx_res = router.ingest_file(xlsx_path)
    assert len(xlsx_res.extracted_tables) == 2
    assert len(xlsx_res.parent_chunks) == 2


def test_router_idempotent_doc_id_and_image_routing(tmp_path: Path):
    """Verifies deterministic SHA-256 doc_id generation and standalone image ingestion."""
    img_path = tmp_path / "chart.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_bytes")

    router = IngestionRouter(
        figure_captioner=lambda p, s: "Bar chart showing latency drop by 4x.",
        figures_dir=tmp_path / "figures",
    )
    res1 = router.ingest_file(img_path)
    res2 = router.ingest_file(img_path)

    assert res1.doc_id == res2.doc_id
    assert res1.file_type == "image"
    assert len(res1.child_chunks) == 1
    assert res1.child_chunks[0].modality == "figure_caption"
    assert "latency drop by 4x" in res1.child_chunks[0].breadcrumb_text
    assert res1.child_chunks[0].image_path is not None


def test_pdf_parser_extracts_sections_tables_and_figures(tmp_path: Path, monkeypatch):
    """Verifies PDFParser reading-order traversal, table DataFrame extraction, and figure cropping."""
    from PIL import Image
    from ingestion.pdf_parser import PDFParser

    class FakeProv:
        def __init__(self, page_no: int):
            self.page_no = page_no

    class FakeItem:
        def __init__(self, label: str, text: str = "", page: int = 1, df=None, img=None):
            self.label = label
            self.text = text
            self.prov = [FakeProv(page)]
            self._df = df
            self._img = img

        def export_to_dataframe(self, doc=None):
            return self._df

        def get_image(self, doc=None):
            return self._img

        def caption_text(self, doc=None):
            return self.text

    class FakeDoc:
        def iterate_items(self):
            df = pd.DataFrame({"Model": ["Qwen3-8B", "Baseline"], "Accuracy": [94.2, 88.1]})
            img = Image.new("RGB", (32, 32), color="blue")
            return [
                (FakeItem("section_header", "1. Introduction", page=1), 1),
                (FakeItem("text", "VeritasAI uses hierarchical parent-child chunking.", page=1), 1),
                (FakeItem("table", page=2, df=df), 1),
                (FakeItem("picture", "Figure 1: MRL Recall Curve", page=3, img=img), 1),
            ]

    class FakeResult:
        document = FakeDoc()

    class FakeConverter:
        def convert(self, _path: str):
            return FakeResult()

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

    parser = PDFParser(figures_dir=tmp_path / "figures")
    monkeypatch.setattr(parser, "_create_converter", lambda: FakeConverter())

    parents, children, tables = parser.parse(pdf_file, doc_id="doc_pdftest99")

    assert len(tables) == 1
    assert list(tables[0].dataframe.columns) == ["model", "accuracy"]
    assert tables[0].page_number == 2

    modalities = [c.modality for c in children]
    assert "text" in modalities
    assert "table_summary" in modalities
    assert "figure_caption" in modalities

    fig_children = [c for c in children if c.modality == "figure_caption"]
    assert len(fig_children) == 1
    assert fig_children[0].image_path is not None
    assert Path(fig_children[0].image_path).exists()

