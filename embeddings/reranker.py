"""Simple BGE cross-encoder reranker using sentence-transformers on MPS GPU."""

import numpy as np
from sentence_transformers import CrossEncoder
import torch


class BGEReranker:
    """Reranks candidate texts using BGE reranker large cross-encoder on MPS GPU."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.model = CrossEncoder(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: list[str], top_k: int = 3) -> list[int]:
        """Returns sorted original text indices from highest to lowest score."""
        if not texts:
            return []

        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices[:top_k].tolist()
