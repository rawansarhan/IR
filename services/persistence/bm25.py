"""حفظ/تحميل نموذج BM25 (مع معاملاته k1, b)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from services.persistence.paths import model_path
from services.retrieval import Bm25Retriever

def save_bm25(dataset: str, retriever: Bm25Retriever, k1: float = 1.5, b: float = 0.75) -> Path:
    path = model_path(dataset, "bm25", k1=k1, b=b)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "model.joblib"
    joblib.dump(
        {"doc_ids": retriever.doc_ids, "k1": retriever.k1, "b": retriever.b, "bm25": retriever._bm25},
        out,
        compress=3,
    )
    return out


def load_bm25(dataset: str, k1: float = 1.5, b: float = 0.75) -> Optional[Bm25Retriever]:
    out = model_path(dataset, "bm25", k1=k1, b=b) / "model.joblib"
    if not out.exists():
        return None
    data = joblib.load(out)
    r = Bm25Retriever(k1=data["k1"], b=data["b"])
    r.doc_ids = data["doc_ids"]
    r._bm25 = data["bm25"]
    return r
