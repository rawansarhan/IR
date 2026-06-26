"""حفظ/تحميل النموذج الهجين المتوازي (Parallel) مع أوزان الدمج."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from services.persistence.paths import model_path
from services.retrieval import HybridParallelRetriever


def save_hybrid_parallel(
    dataset: str,
    retriever: HybridParallelRetriever,
    bw: float = 0.4,
    ew: float = 0.6,
) -> Path:
    path = model_path(dataset, "hybrid_parallel", bw=bw, ew=ew)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "model.pkl"
    with open(out, "wb") as f:
        pickle.dump(
            {
                "bm25_weight": retriever.bm25_weight,
                "emb_weight": retriever.emb_weight,
                "bm25": retriever._bm25,
                "emb": retriever._emb,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return out


def load_hybrid_parallel(bw: float = 0.4, ew: float = 0.6, dataset: str = "") -> Optional[HybridParallelRetriever]:
    out = model_path(dataset, "hybrid_parallel", bw=bw, ew=ew) / "model.pkl"
    if not out.exists():
        return None
    with open(out, "rb") as f:
        data = pickle.load(f)
    r = HybridParallelRetriever(bm25_weight=data["bm25_weight"], emb_weight=data["emb_weight"])
    r._bm25 = data["bm25"]
    r._emb = data["emb"]
    return r
