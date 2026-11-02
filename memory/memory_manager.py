"""3-Tier Memory Manager for week-long multi-turn research conversations."""

import sqlite3
import time
from pathlib import Path
from embeddings.embedder import Embedder
from storage.faiss_store import FAISSStore


class MemoryManager:
    """Manages Working (turns), Episodic (summaries in FAISS), and Semantic (facts) memory."""

    def __init__(
        self,
        db_path: str = "data/db/veritas.sqlite",
        embedder: Embedder | None = None,
        window_size: int = 6,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.window_size = window_size
        self.episode_faiss = FAISSStore(dim=1024, save_dir="data/indexes/memory")
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Creates SQLite tables for Tier 1 turns, Tier 2 episodes, and Tier 3 facts."""
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    compressed INTEGER DEFAULT 0,
                    created_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    episode_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    summary TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    fact TEXT,
                    created_at REAL
                )
                """
            )
            conn.commit()

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Saves a single user or assistant message into Tier 1 working memory."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
            conn.commit()

    def get_recent_turns(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Returns the most recent conversation turns in chronological order."""
        k = limit or self.window_size
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM chat_turns WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
                (session_id, k),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_all_turns(self, session_id: str) -> list[dict]:
        """Returns all conversation turns for UI history restoration."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM chat_turns WHERE session_id = ? ORDER BY turn_id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_fact(self, session_id: str, fact: str) -> None:
        """Stores a persistent research fact or user preference in Tier 3 semantic memory."""
        if not fact.strip():
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO semantic_facts (session_id, fact, created_at) VALUES (?, ?, ?)",
                (session_id, fact.strip(), time.time()),
            )
            conn.commit()

    def get_facts(self, session_id: str, limit: int = 15) -> list[str]:
        """Retrieves pinned research facts for the session."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT fact FROM semantic_facts WHERE session_id = ? ORDER BY fact_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [r["fact"] for r in reversed(rows)]

    def save_episode(self, session_id: str, summary: str) -> str:
        """Saves a compressed episode summary to SQLite and embeds it into episodic FAISS."""
        episode_id = f"ep_{session_id}_{int(time.time() * 1000)}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memory_episodes VALUES (?, ?, ?, ?)",
                (episode_id, session_id, summary, time.time()),
            )
            conn.commit()
        if self.embedder:
            vec = self.embedder.embed_documents([summary])
            self.episode_faiss.add([episode_id], vec)
        return episode_id

    def recall_episodes(self, session_id: str, query: str, top_k: int = 2) -> list[str]:
        """Searches episodic FAISS for relevant past conversation summaries."""
        if not self.embedder or len(self.episode_faiss.ids) == 0:
            return []
        q_vec = self.embedder.embed_query(query)
        hits = self.episode_faiss.search(q_vec, top_k=top_k * 2)
        ep_ids = [eid for eid, _ in hits]
        if not ep_ids:
            return []
        placeholders = ",".join("?" for _ in ep_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT summary FROM memory_episodes WHERE session_id = ? AND episode_id IN ({placeholders})",
                [session_id, *ep_ids],
            ).fetchall()
        return [r["summary"] for r in rows[:top_k]]

    def get_uncompressed_turns(self, session_id: str) -> list[dict]:
        """Fetches turns that have not yet been summarized into an episode."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT turn_id, role, content FROM chat_turns WHERE session_id = ? AND compressed = 0 ORDER BY turn_id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_turns_compressed(self, turn_ids: list[int]) -> None:
        """Marks turns as compressed after summarizing them into an episode."""
        if not turn_ids:
            return
        placeholders = ",".join("?" for _ in turn_ids)
        with self._conn() as conn:
            conn.execute(f"UPDATE chat_turns SET compressed = 1 WHERE turn_id IN ({placeholders})", turn_ids)
            conn.commit()

    def build_memory_context(self, session_id: str, query: str) -> str:
        """Assembles Tier 1 recent turns, Tier 2 recalled episodes, and Tier 3 facts."""
        sections: list[str] = []
        facts = self.get_facts(session_id)
        if facts:
            sections.append("Pinned Research Facts:\n" + "\n".join(f"- {f}" for f in facts))

        episodes = self.recall_episodes(session_id, query)
        if episodes:
            sections.append("Relevant Past Session Summaries:\n" + "\n".join(f"- {ep}" for ep in episodes))

        recent = self.get_recent_turns(session_id)
        if recent:
            turns_str = "\n".join(f"{t['role'].capitalize()}: {t['content'][:400]}" for t in recent)
            sections.append("Recent Conversation History:\n" + turns_str)

        return "\n\n".join(sections)

    def clear(self, session_id: str | None = None) -> None:
        """Clears conversation memory for a specific session or all sessions."""
        with self._conn() as conn:
            if session_id:
                conn.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM memory_episodes WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM semantic_facts WHERE session_id = ?", (session_id,))
            else:
                conn.execute("DELETE FROM chat_turns")
                conn.execute("DELETE FROM memory_episodes")
                conn.execute("DELETE FROM semantic_facts")
                self.episode_faiss.clear()
            conn.commit()
