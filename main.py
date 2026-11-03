"""Entrypoint for VeritasAI RAG pipeline and FastAPI server."""

import sys
from pathlib import Path
import uvicorn
from agent.orchestrator import AgentOrchestrator
from embeddings.embedder import Embedder
from embeddings.reranker import BGEReranker
from ingestion.router import IngestionRouter
from storage.faiss_store import FAISSStore
from storage.sqlite_store import SQLiteStore

BASE_DIR = Path(__file__).resolve().parent


def run_first_cut_demo(sample_file: str | None = None) -> None:
    """Runs a full first-cut demonstration of ingestion, retrieval, and answering."""
    print("=" * 60)
    print("VeritasAI: First-Cut Agentic RAG Demo")
    print("=" * 60)

    # Resolve paths relative to project root
    db_path = BASE_DIR / "data/db/veritas.sqlite"
    index_dir = BASE_DIR / "data/indexes"
    uploads_dir = BASE_DIR / "data/uploads"

    supported_exts = (
        IngestionRouter.PDF_EXTS
        | IngestionRouter.TABULAR_EXTS
        | IngestionRouter.MARKDOWN_EXTS
        | IngestionRouter.IMAGE_EXTS
    )
    if sample_file:
        files_to_ingest = [Path(sample_file)]
    else:
        files_to_ingest = sorted(
            f for f in uploads_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts
        )

    # 1. Initialize components
    print("\n[1/4] Initializing stores and models...")
    sqlite_store = SQLiteStore(str(db_path))
    embedder = Embedder()
    dim = embedder.model.get_sentence_embedding_dimension()
    faiss_store = FAISSStore(dim=dim, save_dir=str(index_dir))
    reranker = BGEReranker()
    orchestrator = AgentOrchestrator(sqlite_store, faiss_store, embedder, reranker)

    # 2. Ingest documents (clear old DB/FAISS first for a clean refresh)
    if not files_to_ingest:
        print(f"No supported files found in {uploads_dir}. Ingestion skipped.")
        return

    print(f"\n[2/4] Clearing old DB/FAISS and freshly ingesting {len(files_to_ingest)} document(s) from {uploads_dir.name}/...")
    sqlite_store.clear()
    faiss_store.clear()
    router = IngestionRouter()
    for file_path in files_to_ingest:
        if not file_path.exists():
            print(f" -> File {file_path} not found, skipping.")
            continue
        print(f" -> Parsing {file_path.name}...")
        try:
            result = router.ingest_file(file_path)
        except Exception as e:
            print(f" -> Failed to parse {file_path.name}: {e}")
            continue

        sqlite_store.save(result.parent_chunks, result.child_chunks, result.extracted_tables)
        if result.child_chunks:
            texts = [c.breadcrumb_text for c in result.child_chunks]
            ids = [c.child_id for c in result.child_chunks]
            vectors = embedder.embed_documents(texts)
            faiss_store.add(ids, vectors)
        print(
            f"    Indexed {file_path.name}: {len(result.parent_chunks)} parents, "
            f"{len(result.child_chunks)} children."
        )

    # 3. Query
    query = "How does deep seek moe architecture look like?"
    print(f"\n[3/4] Running user query: '{query}'...")

    # 4. Agent Execution (FAISS Retrieval -> BGE Reranking -> Answer Generation)
    response = orchestrator.run(query)

    print("\n[4/4] Result:")
    print("-" * 60)
    print("Generated Answer:")
    print(response["answer"])
    print("-" * 60)
    if "latency" in response:
        lat = response["latency"]
        print("Latency Breakdown:")
        print(f" • Query Embedding:     {lat['query_embedding_ms']} ms")
        print(f" • FAISS Search:        {lat['faiss_search_ms']} ms")
        print(f" • SQLite Parent Lookup:{lat['sqlite_lookup_ms']} ms")
        print(f" • BGE Reranking:       {lat['reranking_ms']} ms")
        print(f" • Groq LLM Generation: {lat['llm_generation_ms']} ms")
        print(f" • Total Latency:       {lat['total_ms']} ms ({lat['total_ms']/1000:.2f} s)")
        print("-" * 60)
    print("Top Reranked Sources:")
    for idx, src in enumerate(response["sources"], 1):
        print(f" {idx}. [Doc: {src['doc']} | Section: {src['section']} | Page: {src['page']}]")
        print(f"    Snippet: {src['content'][:150]}...\n")
    print("=" * 60)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        print("Starting VeritasAI API server at http://0.0.0.0:8000 ...")
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["agent", "api", "config", "embeddings", "ingestion", "memory", "storage"],
        )
    else:
        run_first_cut_demo()
