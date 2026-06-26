"""FAISS Vector Store — dense vector index for fast semantic search.

الميزة الإضافية: Vector Store
─────────────────────────────
بدل البحث الدلالي بضرب مصفوفة كامل (query @ all_doc_embeddings.T) — وهو بطيء
على مئات آلاف الوثائق — نخزّن الـ embeddings في فهرس FAISS متخصص.

FAISS IndexFlatIP:
    - Inner Product على vectors مُطبّعة (normalized) = Cosine Similarity
    - بحث دقيق (exact) وسريع جدًا
    - يُحفظ على القرص ويُحمّل بسرعة (.faiss)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
VECTOR_DIR = DATA_ROOT / "vector_store"


class FaissVectorStore:
    """FAISS-based dense vector store with cosine similarity search."""

    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim
        self.index = None
        self.doc_ids: List[str] = []

    # ── Build ─────────────────────────────────────────────────────────
    def build(self, doc_ids: List[str], embeddings: np.ndarray) -> None:
        """Build FAISS index from pre-computed (normalized) embeddings."""
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        # safety: normalize so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        self.dim = embeddings.shape[1]
        self.doc_ids = list(doc_ids)
        # IndexFlatIP = exact inner-product (cosine on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    # ── Search ────────────────────────────────────────────────────────
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Vector store is not built/loaded yet.")

        q = np.ascontiguousarray(query_embedding.astype("float32")).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, dataset: str) -> Path:
        base = VECTOR_DIR / dataset
        base.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(base / "index.faiss"))
        with open(base / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"doc_ids": self.doc_ids, "dim": self.dim}, f)
        return base

    @classmethod
    def load(cls, dataset: str) -> Optional["FaissVectorStore"]:
        base = VECTOR_DIR / dataset
        index_file = base / "index.faiss"
        meta_file = base / "meta.json"
        if not index_file.exists() or not meta_file.exists():
            return None
        store = cls()
        store.index = faiss.read_index(str(index_file))
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        store.doc_ids = meta["doc_ids"]
        store.dim = meta["dim"]
        return store

    @staticmethod
    def exists(dataset: str) -> bool:
        base = VECTOR_DIR / dataset
        return (base / "index.faiss").exists() and (base / "meta.json").exists()

    def stats(self) -> dict:
        return {
            "num_vectors": int(self.index.ntotal) if self.index else 0,
            "dimension": self.dim,
            "index_type": "IndexFlatIP (cosine)",
            "backend": "FAISS",
        }
