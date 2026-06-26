"""حفظ/تحميل النموذج الهجين التسلسلي (Serial)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from services.persistence.paths import model_path
from services.retrieval import HybridSerialRetriever


def save_hybrid_serial(dataset: str, retriever: HybridSerialRetriever) -> Path:
    path = model_path(dataset, "hybrid_serial")
    path.mkdir(parents=True, exist_ok=True)
    out = path / "model.pkl"
    with open(out, "wb") as f:
        pickle.dump(
            {
                "candidate_k": retriever.candidate_k,
                "bm25": retriever._bm25,
                "emb": retriever._emb,
                "doc_text_map": retriever._doc_text_map,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return out


def load_hybrid_serial(dataset: str) -> Optional[HybridSerialRetriever]:
    out = model_path(dataset, "hybrid_serial") / "model.pkl"
    if not out.exists():
        return None
    with open(out, "rb") as f:
        data = pickle.load(f)
    r = HybridSerialRetriever(candidate_k=data["candidate_k"])
    r._bm25 = data["bm25"]
    r._emb = data["emb"]
    r._doc_text_map = data["doc_text_map"]
    return r
