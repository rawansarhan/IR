"""حفظ/تحميل نموذج TF-IDF."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from services.persistence.paths import model_path
from services.retrieval import TfidfRetriever


def save_tfidf(dataset: str, retriever: TfidfRetriever) -> Path:
    path = model_path(dataset, "tfidf")
    path.mkdir(parents=True, exist_ok=True)
    out = path / "model.joblib"
    joblib.dump(
        {"doc_ids": retriever.doc_ids, "vectorizer": retriever.vectorizer, "doc_matrix": retriever.doc_matrix},
        out,
        compress=3,
    )
    return out


def load_tfidf(dataset: str) -> Optional[TfidfRetriever]:
    out = model_path(dataset, "tfidf") / "model.joblib"
    if not out.exists():
        return None
    data = joblib.load(out)
    r = TfidfRetriever()
    r.doc_ids = data["doc_ids"]
    r.vectorizer = data["vectorizer"]
    r.doc_matrix = data["doc_matrix"]
    return r
