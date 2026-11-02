"""Simple vector index for searching vector embeddings."""

import json
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FAISSStore:
    """Stores embeddings in FAISS (or numpy) and maps them to chunk IDs."""

    def __init__(self, dim: int = 1024, save_dir: str = "data/indexes") -> None:
        self.dim = dim
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.save_dir / "faiss.index"
        self.ids_file = self.save_dir / "ids.json"
        self.vecs_file = self.save_dir / "vecs.npy"

        self.ids: list[str] = []
        self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dim) if faiss else None
        self.load()

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Adds chunk IDs and their vector embeddings to the index."""
        if len(chunk_ids) == 0:
            return
        vecs = np.array(vectors, dtype=np.float32)
        if self.index is not None:
            self.index.add(vecs)
        self.vectors = np.vstack([self.vectors, vecs])
        self.ids.extend(chunk_ids)
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Searches index for nearest chunk IDs to the query vector."""
        if len(self.ids) == 0:
            return []
        q_vec = np.array([query_vector], dtype=np.float32)
        top_k = min(top_k, len(self.ids))

        if self.index is not None:
            scores, indices = self.index.search(q_vec, top_k)
            return [(self.ids[idx], float(s)) for idx, s in zip(indices[0], scores[0]) if idx >= 0]

        # Pure numpy dot-product fallback (equivalent to IndexFlatIP)
        scores = np.dot(self.vectors, q_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([scores])
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in best_indices]

    def save(self) -> None:
        """Saves index and chunk IDs to disk."""
        self.ids_file.write_text(json.dumps(self.ids))
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        else:
            np.save(str(self.vecs_file), self.vectors)

    def load(self) -> None:
        """Loads index and chunk IDs from disk if they exist."""
        if self.ids_file.exists():
            self.ids = json.loads(self.ids_file.read_text())
        if self.index is not None and self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
        elif self.vecs_file.exists():
            self.vectors = np.load(str(self.vecs_file))

    def clear(self) -> None:
        """Clears all vectors and IDs from memory and disk."""
        self.ids = []
        self.vectors = np.empty((0, self.dim), dtype=np.float32)
        if self.index is not None and faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
        for f in [self.ids_file, self.vecs_file, self.index_file]:
            if f.exists():
                f.unlink()
