"""Simple Qwen embedding generator using sentence-transformers on MPS GPU."""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """Creates Qwen vector embeddings for documents and research queries."""

    INSTRUCTION = (
        "Instruct: Given a research query, retrieve relevant passages, tables, or chart descriptions that answer the query\n"
        "Query: "
    )

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(
                self.model_name, device=device, local_files_only=True, model_kwargs={"dtype": torch.float32}
            )
        except Exception:
            self.model = SentenceTransformer(
                self.model_name, device=device, model_kwargs={"dtype": torch.float32}
            )

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a query with the research retrieval instruction."""
        text = f"{self.INSTRUCTION}{query}"
        return self.model.encode(text, normalize_embeddings=True)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embeds document chunk texts directly without instruction."""
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True)
