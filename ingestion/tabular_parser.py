"""Excel and CSV parser for dual indexing (FAISS discovery summary + SQLite execution table)."""

import json
import re
from pathlib import Path
import pandas as pd
from ingestion.chunker import HierarchicalChunker
from ingestion.models import ChildChunk, ExtractedTable, ParentChunk


class TabularParser:
    """Extracts structured DataFrames and semantic discovery summaries from Excel/CSV files."""

    def __init__(self, chunker: HierarchicalChunker | None = None) -> None:
        self.chunker = chunker or HierarchicalChunker()

    def parse(
        self, file_path: Path, doc_id: str
    ) -> tuple[list[ParentChunk], list[ChildChunk], list[ExtractedTable]]:
        """Parses all sheets of an Excel workbook or a CSV file into tables and chunks."""
        suffix = file_path.suffix.lower()
        sheets: dict[str, pd.DataFrame] = {}

        if suffix == ".csv":
            sheets[file_path.stem] = pd.read_csv(file_path)
        elif suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported tabular format: {suffix}")

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        tables: list[ExtractedTable] = []

        for idx, (sheet_name, df) in enumerate(sheets.items()):
            clean_df = self._sanitize_dataframe(df)
            if clean_df.empty:
                continue

            table_id = f"tbl_{doc_id[:8]}_{idx}"
            sql_name = self._make_sql_table_name(file_path.stem, str(sheet_name), doc_id, idx)
            schema_dict = {col: str(dtype) for col, dtype in clean_df.dtypes.items()}
            summary_text = self._build_table_summary(
                filename=file_path.name,
                sheet_name=str(sheet_name),
                sql_table_name=sql_name,
                df=clean_df,
                schema_dict=schema_dict,
            )

            tables.append(
                ExtractedTable(
                    table_id=table_id,
                    doc_id=doc_id,
                    sql_table_name=sql_name,
                    dataframe=clean_df,
                    schema_json=json.dumps(schema_dict),
                    summary_text=summary_text,
                    page_number=idx + 1,
                )
            )

            p_chunks, c_chunks = self.chunker.chunk_section(
                doc_id=doc_id,
                filename=file_path.name,
                section_path=f"Sheet: {sheet_name}",
                page_number=idx + 1,
                text=summary_text,
                modality="table_summary",
                start_index=idx * 100,
            )
            parents.extend(p_chunks)
            children.extend(c_chunks)

        return parents, children, tables

    @staticmethod
    def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Drops empty rows/columns and normalizes column headers into valid SQL identifiers."""
        cleaned = df.dropna(how="all").dropna(axis=1, how="all").copy()
        if cleaned.empty:
            return cleaned

        seen: dict[str, int] = {}
        new_cols: list[str] = []
        for i, col in enumerate(cleaned.columns):
            base = re.sub(r"\W+", "_", str(col).strip().lower()).strip("_")
            if not base or base[0].isdigit():
                base = f"col_{base or i}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            new_cols.append(f"{base}_{count}" if count else base)

        cleaned.columns = new_cols
        return cleaned

    @staticmethod
    def _make_sql_table_name(stem: str, sheet: str, doc_id: str, idx: int) -> str:
        safe_stem = re.sub(r"\W+", "_", stem.lower()).strip("_")[:20]
        safe_sheet = re.sub(r"\W+", "_", sheet.lower()).strip("_")[:20]
        return f"t_{safe_stem}_{safe_sheet}_{doc_id[:6]}_{idx}"

    @staticmethod
    def _format_markdown_sample(df: pd.DataFrame, max_rows: int = 3) -> str:
        """Formats sample DataFrame rows as a clean Markdown table without extra dependencies."""
        sample = df.head(max_rows)
        cols = [str(c) for c in sample.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = [
            "| " + " | ".join(str(val) for val in row) + " |"
            for row in sample.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *body])

    @classmethod
    def _build_table_summary(
        cls,
        filename: str,
        sheet_name: str,
        sql_table_name: str,
        df: pd.DataFrame,
        schema_dict: dict[str, str],
    ) -> str:
        """Generates a rich natural-language and markdown summary of a table for vector retrieval."""
        rows, cols = df.shape
        col_list = ", ".join(f"{c} ({t})" for c, t in schema_dict.items())
        sample_md = cls._format_markdown_sample(df, max_rows=3)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        stats_lines: list[str] = []
        for col in numeric_cols[:8]:
            series = df[col].dropna()
            if not series.empty:
                stats_lines.append(
                    f"- {col}: min={series.min():.4g}, max={series.max():.4g}, mean={series.mean():.4g}"
                )

        stats_block = "\n".join(stats_lines) if stats_lines else "No numeric columns."
        return (
            f"Table Summary for '{filename}' (Sheet/Section: '{sheet_name}')\n"
            f"SQL Table Name: {sql_table_name}\n"
            f"Dimensions: {rows} rows x {cols} columns\n"
            f"Columns & Types: {col_list}\n"
            f"Numeric Highlights:\n{stats_block}\n"
            f"Sample Rows:\n{sample_md}"
        )
