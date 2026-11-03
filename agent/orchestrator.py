"""Simple pure-Python agentic RAG orchestrator with Groq generation and latency tracking."""

import os
import time
from dotenv import load_dotenv
import httpx
from embeddings.embedder import Embedder
from embeddings.reranker import BGEReranker
from memory.memory_manager import MemoryManager
from storage.faiss_store import FAISSStore
from storage.sqlite_store import SQLiteStore

load_dotenv()


class AgentOrchestrator:
    """Coordinates memory recall, retrieval, reranking, and Groq LLM generation."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        faiss_store: FAISSStore,
        embedder: Embedder,
        reranker: BGEReranker | None = None,
        memory: MemoryManager | None = None,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
    ) -> None:
        self.sqlite = sqlite_store
        self.faiss = faiss_store
        self.embedder = embedder
        self.reranker = reranker or BGEReranker()
        self.memory = memory or MemoryManager(str(sqlite_store.db_path), embedder)
        self.model = model
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.api_key = api_key if api_key is not None else self.groq_key

    def retrieve_with_latency(self, query: str, top_k: int = 3) -> tuple[list[dict], dict[str, float]]:
        """Retrieves chunks and tracks latency for embedding, FAISS, SQLite, and reranking."""
        timings: dict[str, float] = {}

        # 1. Query Embedding
        t_start = time.perf_counter()
        query_vec = self.embedder.embed_query(query)
        timings["query_embedding_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        # 2. FAISS Vector Search
        t_faiss = time.perf_counter()
        candidates = self.faiss.search(query_vec, top_k=top_k * 3)
        timings["faiss_search_ms"] = round((time.perf_counter() - t_faiss) * 1000, 2)

        if not candidates:
            timings["sqlite_lookup_ms"] = 0.0
            timings["reranking_ms"] = 0.0
            return [], timings

        # 3. SQLite Lookup
        t_sqlite = time.perf_counter()
        child_ids = [cid for cid, _ in candidates]
        id_map = self.sqlite.get_child_to_parent_map(child_ids)
        all_parent_ids = list(dict.fromkeys(id_map.values()))
        parents_by_id = {p.parent_id: p for p in self.sqlite.get_parent_chunks(all_parent_ids)}
        timings["sqlite_lookup_ms"] = round((time.perf_counter() - t_sqlite) * 1000, 2)

        # 4. BGE Cross-Encoder Reranking
        t_rerank = time.perf_counter()
        parent_texts = [parents_by_id[pid].content for pid in all_parent_ids if pid in parents_by_id]
        best_indices = self.reranker.rerank(query, parent_texts, top_k=top_k)
        timings["reranking_ms"] = round((time.perf_counter() - t_rerank) * 1000, 2)

        chosen_parents = [parents_by_id[all_parent_ids[i]] for i in best_indices if i < len(all_parent_ids)]
        sources = [
            {
                "doc": p.filename,
                "section": p.section_path,
                "page": p.page_number,
                "content": p.content,
            }
            for p in chosen_parents
        ]
        return sources, timings

    def call_groq(self, prompt: str, system: str) -> tuple[str, float]:
        """Calls Groq openai/gpt-oss-120b text model and returns (answer, latency_ms)."""
        t_start = time.perf_counter()
        if not self.api_key:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return f"[No GROQ_API_KEY] Prompt: {prompt}", latency_ms

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        answer = resp.json()["choices"][0]["message"]["content"]
        return answer, latency_ms

    def _update_memory(self, session_id: str, query: str, answer: str) -> None:
        """Saves the current turn and compresses older turns into episodes and facts every 6 turns."""
        self.memory.save_turn(session_id, "user", query)
        self.memory.save_turn(session_id, "assistant", answer)

        uncompressed = self.memory.get_uncompressed_turns(session_id)
        if len(uncompressed) >= 6:
            to_compress = uncompressed[:6]
            convo_text = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in to_compress)
            if self.api_key:
                try:
                    sys_prompt = (
                        "Summarize this research conversation segment in 2-3 sentences on the first line. "
                        "Then after 'FACTS:', list up to 2 key research facts or user preferences (one per line)."
                    )
                    summary_raw, _ = self.call_groq(convo_text, sys_prompt)
                    parts = summary_raw.split("FACTS:")
                    episode_summary = parts[0].strip()
                    if episode_summary:
                        self.memory.save_episode(session_id, episode_summary)
                    if len(parts) > 1:
                        for line in parts[1].strip().splitlines():
                            cleaned = line.lstrip("-•* ").strip()
                            if cleaned:
                                self.memory.add_fact(session_id, cleaned)
                except Exception:
                    pass
            self.memory.mark_turns_compressed([t["turn_id"] for t in to_compress])

    def run(self, query: str, session_id: str = "default") -> dict:
        """Runs the agent pipeline with 3-tier memory, logs latency breakdown, and returns answer."""
        t_total_start = time.perf_counter()

        # 0. Memory Recall (Working + Episodic + Semantic Facts)
        t_mem = time.perf_counter()
        memory_context = self.memory.build_memory_context(session_id, query)
        recent_turns = self.memory.get_recent_turns(session_id, limit=2)
        search_query = query
        if recent_turns and len(query.split()) < 10:
            last_user = next((t["content"] for t in reversed(recent_turns) if t["role"] == "user"), "")
            if last_user:
                search_query = f"{last_user} {query}"
        memory_recall_ms = round((time.perf_counter() - t_mem) * 1000, 2)

        chunks, timings = self.retrieve_with_latency(search_query)
        timings["memory_recall_ms"] = memory_recall_ms

        if not chunks:
            system = "You are a helpful research assistant. Use conversation memory if relevant."
            prompt = (
                f"Answer the user query only from the given memory context: {memory_context}\n\nQuestion: {query}".strip()
                if memory_context
                else query
            )
            answer, llm_ms = self.call_groq(prompt, system)
            self._update_memory(session_id, query, answer)
            timings["llm_generation_ms"] = llm_ms
            timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
            return {
                "query": query,
                "session_id": session_id,
                "answer": answer,
                "sources": [],
                "latency": timings,
            }

        context = "\n\n".join(
            [f"[Doc: {c['doc']} | Page: {c['page']} | Section: {c['section']}]\n{c['content']}" for c in chunks]
        )
        system = (
            "You are a research assistant in a week-long multi-turn study. Answer the user query "
            "using the provided document context and conversation memory. "
            "Cite document sources with [Doc, Page]. If the context contains a figure image link "
            "like ![...](/figures/...), include that markdown image link in your answer when relevant."
        )
        mem_block = f"Conversation Memory:\n{memory_context}\n\n" if memory_context else ""
        prompt = f"{mem_block}Document Context:\n{context}\n\nQuestion: {query}"

        answer, llm_ms = self.call_groq(prompt, system)
        self._update_memory(session_id, query, answer)
        timings["llm_generation_ms"] = llm_ms
        timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)

        # Print latency breakdown to console
        print("\n" + "=" * 45)
        print("          LATENCY BREAKDOWN          ")
        print("=" * 45)
        print(f" • Memory Recall:       {timings['memory_recall_ms']:>8.2f} ms")
        print(f" • Query Embedding:     {timings['query_embedding_ms']:>8.2f} ms")
        print(f" • FAISS Search:        {timings['faiss_search_ms']:>8.2f} ms")
        print(f" • SQLite Parent Lookup:{timings['sqlite_lookup_ms']:>8.2f} ms")
        print(f" • BGE Reranking:       {timings['reranking_ms']:>8.2f} ms")
        print(f" • Groq LLM Generation: {timings['llm_generation_ms']:>8.2f} ms")
        print("-" * 45)
        print(f" Total Query Latency:   {timings['total_ms']:>8.2f} ms ({timings['total_ms']/1000:.2f} s)")
        print("=" * 45 + "\n")

        return {
            "query": query,
            "session_id": session_id,
            "answer": answer,
            "sources": chunks,
            "latency": timings,
        }
