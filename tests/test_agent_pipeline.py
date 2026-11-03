"""Tests for SQLiteStore, FAISSStore, QwenReranker, and AgentOrchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from agent.orchestrator import AgentOrchestrator
from embeddings.embedder import Embedder
from embeddings.reranker import BGEReranker
from ingestion.models import ChildChunk, ParentChunk
from storage.faiss_store import FAISSStore
from storage.sqlite_store import SQLiteStore


def test_sqlite_store_save_and_retrieve(tmp_path: Path):
    """Tests saving and retrieving parent and child chunks in SQLite."""
    db_file = tmp_path / "test.sqlite"
    store = SQLiteStore(str(db_file))

    parents = [
        ParentChunk(
            parent_id="p1",
            doc_id="doc1",
            filename="paper.pdf",
            section_path="Intro",
            page_number=1,
            content="This is the introduction about transformers.",
        )
    ]
    children = [
        ChildChunk(
            child_id="c1",
            parent_id="p1",
            doc_id="doc1",
            breadcrumb_text="[Doc: paper.pdf | Page: 1] intro chunk",
            modality="text",
            page_number=1,
        )
    ]

    store.save(parents, children)

    retrieved = store.get_parent_chunks(["p1"])
    assert len(retrieved) == 1
    assert retrieved[0].parent_id == "p1"
    assert "transformers" in retrieved[0].content

    id_map = store.get_child_to_parent_map(["c1"])
    assert id_map["c1"] == "p1"


def test_faiss_store_add_and_search(tmp_path: Path):
    """Tests adding vectors to FAISS and searching for nearest matches."""
    store = FAISSStore(dim=4, save_dir=str(tmp_path / "indexes"))

    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    store.add(["c1", "c2"], vectors)

    query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=1)

    assert len(results) == 1
    assert results[0][0] == "c1"


def test_embedder_query_instruction():
    """Tests that embedder adds query instruction prefix."""
    with patch("embeddings.embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        embedder = Embedder(model_name="dummy")

        embedder.embed_query("What is attention?")
        mock_model.encode.assert_called_once()
        call_arg = mock_model.encode.call_args[0][0]
        assert "Instruct: Given a research query" in call_arg
        assert "What is attention?" in call_arg


def test_bge_reranker_mock():
    """Tests that BGEReranker sorts text indices by score."""
    with patch("embeddings.reranker.CrossEncoder") as mock_ce:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.9, 0.5])
        mock_ce.return_value = mock_model

        reranker = BGEReranker(model_name="dummy")
        best_indices = reranker.rerank("query", ["doc0", "doc1", "doc2"], top_k=2)
        # doc1 (0.9) is highest, doc2 (0.5) is second
        assert best_indices == [1, 2]


def test_agent_orchestrator_end_to_end_with_rerank(tmp_path: Path):
    """Tests the full retrieval, reranking, and answering flow in AgentOrchestrator."""
    db_file = tmp_path / "test.sqlite"
    sqlite = SQLiteStore(str(db_file))
    faiss_store = FAISSStore(dim=4, save_dir=str(tmp_path / "indexes"))

    parents = [
        ParentChunk(
            parent_id="p1",
            doc_id="doc1",
            filename="attention.pdf",
            section_path="Abstract",
            page_number=1,
            content="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        )
    ]
    children = [
        ChildChunk(
            child_id="c1",
            parent_id="p1",
            doc_id="doc1",
            breadcrumb_text="[Doc: attention.pdf | Page: 1] sequence transduction models",
            modality="text",
            page_number=1,
        )
    ]
    sqlite.save(parents, children)

    chunk_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    faiss_store.add(["c1"], chunk_vec)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [0]

    orchestrator = AgentOrchestrator(
        sqlite_store=sqlite,
        faiss_store=faiss_store,
        embedder=mock_embedder,
        reranker=mock_reranker,
        api_key="",  # Runs offline mode
    )

    result = orchestrator.run("What are sequence models based on?")
    assert "query" in result
    assert "answer" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc"] == "attention.pdf"
    mock_reranker.rerank.assert_called_once()
