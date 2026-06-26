"""حفظ/تحميل نموذج Embedding (مصفوفة المتجهات كـ npz مضغوط)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from services.persistence.paths import model_path
from services.retrieval import EmbeddingRetriever


def save_embedding(dataset: str, retriever: EmbeddingRetriever) -> Path:
    path = model_path(dataset, "embedding")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "embeddings.npz", embeddings=retriever.doc_embeddings)
    meta = {"doc_ids": retriever.doc_ids, "model_name": retriever.model_name}
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return path


def load_embedding(dataset: str) -> Optional[EmbeddingRetriever]:
    base = model_path(dataset, "embedding")
    emb_file = base / "embeddings.npz"
    meta_file = base / "meta.json"
    if not emb_file.exists() or not meta_file.exists():
        return None
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    r = EmbeddingRetriever(model_name=meta["model_name"])
    r.doc_ids = meta["doc_ids"]
    r.doc_embeddings = np.load(emb_file)["embeddings"]
    return r
