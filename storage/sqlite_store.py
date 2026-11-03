"""Simple SQLite store for parent chunks, child chunks, and tables."""

from pathlib import Path
import sqlite3
from ingestion.models import ChildChunk, ExtractedTable, ParentChunk


class SQLiteStore:
    """Stores text chunks and tables in SQLite."""

    def __init__(self, db_path: str = "data/db/veritas.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables in SQLite."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    filename TEXT,
                    section_path TEXT,
                    page_number INTEGER,
                    content TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS child_chunks (
                    child_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    doc_id TEXT,
                    breadcrumb_text TEXT
                )
                """
            )
            conn.commit()

    def save(
        self,
        parents: list[ParentChunk],
        children: list[ChildChunk],
        tables: list[ExtractedTable] | None = None,
    ) -> None:
        """Saves parent chunks, child chunks, and tables to SQLite."""
        with self._get_connection() as conn:
            for p in parents:
                conn.execute(
                    "INSERT OR REPLACE INTO parent_chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (p.parent_id, p.doc_id, p.filename, p.section_path, p.page_number, p.content),
                )
            for c in children:
                conn.execute(
                    "INSERT OR REPLACE INTO child_chunks VALUES (?, ?, ?, ?)",
                    (c.child_id, c.parent_id, c.doc_id, c.breadcrumb_text),
                )
            if tables:
                for t in tables:
                    if t.dataframe is not None and not t.dataframe.empty:
                        t.dataframe.to_sql(t.sql_table_name, conn, if_exists="replace", index=False)
            conn.commit()

    def get_parent_chunks(self, parent_ids: list[str]) -> list[ParentChunk]:
        """Gets ParentChunk objects for a list of parent_ids."""
        if not parent_ids:
            return []
        placeholders = ",".join("?" for _ in parent_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM parent_chunks WHERE parent_id IN ({placeholders})",
                parent_ids,
            )
            rows = cursor.fetchall()
            return [
                ParentChunk(
                    parent_id=r["parent_id"],
                    doc_id=r["doc_id"],
                    filename=r["filename"],
                    section_path=r["section_path"],
                    page_number=r["page_number"],
                    content=r["content"],
                )
                for r in rows
            ]

    def get_child_to_parent_map(self, child_ids: list[str]) -> dict[str, str]:
        """Returns a mapping from child_id to parent_id."""
        if not child_ids:
            return {}
        placeholders = ",".join("?" for _ in child_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT child_id, parent_id FROM child_chunks WHERE child_id IN ({placeholders})",
                child_ids,
            )
            return {r["child_id"]: r["parent_id"] for r in cursor.fetchall()}

    def execute_sql(self, query: str) -> list[dict]:
        """Runs a read-only SQL query on the database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(r) for r in cursor.fetchall()]

    def clear(self) -> None:
        """Clears all stored chunks and tables."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM parent_chunks")
            conn.execute("DELETE FROM child_chunks")
            conn.commit()
