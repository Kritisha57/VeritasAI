"""PDF parser powered by Granite-Docling for layout, table, formula, and chart extraction."""

import gc
import json
from pathlib import Path
from typing import Callable
from config.settings import settings
from ingestion.chunker import HierarchicalChunker
from ingestion.models import ChildChunk, ExtractedTable, ParentChunk
from ingestion.tabular_parser import TabularParser


class PDFParser:
    """Extracts structured text, DataFrames, and cropped charts from PDFs using Docling."""

    def __init__(
        self,
        chunker: HierarchicalChunker | None = None,
        tabular_parser: TabularParser | None = None,
        figure_captioner: Callable[[Path, str], str] | None = None,
        figures_dir: Path = settings.figures_dir,
    ) -> None:
        self.chunker = chunker or HierarchicalChunker()
        self.tabular = tabular_parser or TabularParser(self.chunker)
        self.figure_captioner = figure_captioner
        self.figures_dir = figures_dir

    def parse(
        self, file_path: Path, doc_id: str
    ) -> tuple[list[ParentChunk], list[ChildChunk], list[ExtractedTable]]:
        """Runs Docling conversion, extracts text/tables/figures, and frees model memory."""
        converter = self._create_converter()
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        tables: list[ExtractedTable] = []

        current_section = "Document Root"
        current_page = 1
        text_buffer: list[str] = []
        section_counter = 0
        table_counter = 0
        figure_counter = 0

        def flush_text_buffer() -> None:
            nonlocal section_counter
            joined = "\n\n".join(text_buffer).strip()
            text_buffer.clear()
            if not joined:
                return
            p_list, c_list = self.chunker.chunk_section(
                doc_id=doc_id,
                filename=file_path.name,
                section_path=current_section,
                page_number=current_page,
                text=joined,
                modality="text",
                start_index=section_counter * 100,
            )
            parents.extend(p_list)
            children.extend(c_list)
            section_counter += 1

        for item, _level in doc.iterate_items():
            page_no = self._get_item_page(item, default=current_page)
            label = getattr(item, "label", "")
            label_str = str(label).lower()

            if "section_header" in label_str or "title" in label_str:
                flush_text_buffer()
                header_text = getattr(item, "text", "").strip()
                if header_text:
                    current_section = header_text
                current_page = page_no

            elif "table" in label_str and hasattr(item, "export_to_dataframe"):
                flush_text_buffer()
                current_page = page_no
                p_tbl, c_tbl, ext_tbl = self._process_table_item(
                    item, doc, file_path.name, doc_id, current_section, current_page, table_counter
                )
                if ext_tbl is not None:
                    tables.append(ext_tbl)
                    parents.extend(p_tbl)
                    children.extend(c_tbl)
                    table_counter += 1

            elif "picture" in label_str or "figure" in label_str or "chart" in label_str:
                flush_text_buffer()
                current_page = page_no
                p_fig, c_fig = self._process_figure_item(
                    item, doc, file_path.name, doc_id, current_section, current_page, figure_counter
                )
                parents.extend(p_fig)
                children.extend(c_fig)
                figure_counter += 1

            else:
                item_text = getattr(item, "text", "").strip()
                if item_text:
                    if page_no != current_page and text_buffer:
                        flush_text_buffer()
                    current_page = page_no
                    text_buffer.append(item_text)

        flush_text_buffer()

        # Explicitly release Docling pipeline memory on 8GB Apple M1
        del conv_result
        del converter
        gc.collect()

        return parents, children, tables

    @staticmethod
    def _create_converter():
        """Builds a Docling DocumentConverter with table and figure image extraction enabled."""
        import os
        import certifi
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = settings.docling_do_ocr
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = settings.docling_image_scale

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    @staticmethod
    def _get_item_page(item, default: int = 1) -> int:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0 and hasattr(prov[0], "page_no"):
            return int(prov[0].page_no)
        return default

    def _process_table_item(
        self,
        item,
        doc,
        filename: str,
        doc_id: str,
        section: str,
        page: int,
        idx: int,
    ) -> tuple[list[ParentChunk], list[ChildChunk], ExtractedTable | None]:
        """Converts a Docling TableItem into both a SQLite DataFrame and FAISS summary chunks."""
        try:
            df = item.export_to_dataframe(doc=doc)
        except TypeError:
            df = item.export_to_dataframe()

        clean_df = self.tabular._sanitize_dataframe(df)
        if clean_df.empty:
            return [], [], None

        table_id = f"pdftbl_{doc_id[:8]}_{idx}"
        sql_name = self.tabular._make_sql_table_name(Path(filename).stem, f"p{page}_t{idx}", doc_id, idx)
        schema_dict = {col: str(dtype) for col, dtype in clean_df.dtypes.items()}
        summary_text = self.tabular._build_table_summary(
            filename=filename,
            sheet_name=f"{section} (Page {page}, Table {idx + 1})",
            sql_table_name=sql_name,
            df=clean_df,
            schema_dict=schema_dict,
        )

        ext_table = ExtractedTable(
            table_id=table_id,
            doc_id=doc_id,
            sql_table_name=sql_name,
            dataframe=clean_df,
            schema_json=json.dumps(schema_dict),
            summary_text=summary_text,
            page_number=page,
        )
        p_chunks, c_chunks = self.chunker.chunk_section(
            doc_id=doc_id,
            filename=filename,
            section_path=f"{section} > Table {idx + 1}",
            page_number=page,
            text=summary_text,
            modality="table_summary",
            start_index=5000 + idx * 10,
        )
        return p_chunks, c_chunks, ext_table

    def _process_figure_item(
        self,
        item,
        doc,
        filename: str,
        doc_id: str,
        section: str,
        page: int,
        idx: int,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Saves cropped figure image to disk and indexes its caption/VLM description."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        img_path: Path | None = None

        if hasattr(item, "get_image"):
            pil_img = item.get_image(doc)
            if pil_img is not None:
                img_path = self.figures_dir / f"{doc_id[:8]}_p{page}_fig{idx}.png"
                pil_img.save(img_path, format="PNG")

        caption = ""
        if hasattr(item, "caption_text"):
            try:
                caption = item.caption_text(doc).strip()
            except Exception:
                caption = ""

        if self.figure_captioner and img_path and img_path.exists():
            vlm_desc = self.figure_captioner(img_path, caption)
            caption = f"{caption}\n{vlm_desc}".strip()

        if not caption:
            caption = f"Figure {idx + 1} in section '{section}' on page {page} of {filename}."

        img_md = f"\n![Figure {idx + 1}](/figures/{img_path.name})" if img_path else ""
        fig_text = f"[Figure {idx + 1} | Image: {img_path.name if img_path else 'N/A'}] {caption}{img_md}"
        p_chunks, c_chunks = self.chunker.chunk_section(
            doc_id=doc_id,
            filename=filename,
            section_path=f"{section} > Figure {idx + 1}",
            page_number=page,
            text=fig_text,
            modality="figure_caption",
            start_index=8000 + idx * 10,
        )
        for child in c_chunks:
            child.image_path = str(img_path) if img_path else None
        return p_chunks, c_chunks
