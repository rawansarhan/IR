"""الواجهة الموحّدة: اختيار النموذج الصحيح للحفظ/التحميل + بناء كل النماذج دفعة واحدة."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from services.persistence.bm25 import load_bm25, save_bm25
from services.persistence.embedding import load_embedding, save_embedding
from services.persistence.hybrid_parallel import load_hybrid_parallel, save_hybrid_parallel
from services.persistence.hybrid_serial import load_hybrid_serial, save_hybrid_serial
from services.persistence.paths import model_path
from services.persistence.tfidf import load_tfidf, save_tfidf

# النماذج الافتراضية التي تُبنى عند /load
DEFAULT_MODELS = [
    ("tfidf", {}),
    ("bm25", {"k1": 1.5, "b": 0.75}),
    ("embedding", {}),
    ("hybrid_serial", {}),
    ("hybrid_parallel", {"bw": 0.4, "ew": 0.6}),
]


def save_retriever(dataset: str, model: str, retriever: object, **params) -> Path:
    if model == "tfidf":
        return save_tfidf(dataset, retriever)
    if model == "bm25":
        return save_bm25(dataset, retriever, k1=params.get("k1", 1.5), b=params.get("b", 0.75))
    if model == "embedding":
        return save_embedding(dataset, retriever)
    if model == "hybrid_serial":
        return save_hybrid_serial(dataset, retriever)
    if model == "hybrid_parallel":
        return save_hybrid_parallel(dataset, retriever, bw=params.get("bw", 0.4), ew=params.get("ew", 0.6))
    raise ValueError(f"Unknown model: {model}")


def load_retriever(dataset: str, model: str, **params) -> Optional[object]:
    if model == "tfidf":
        return load_tfidf(dataset)
    if model == "bm25":
        return load_bm25(dataset, k1=params.get("k1", 1.5), b=params.get("b", 0.75))
    if model == "embedding":
        return load_embedding(dataset)
    if model == "hybrid_serial":
        return load_hybrid_serial(dataset)
    if model == "hybrid_parallel":
        return load_hybrid_parallel(dataset=dataset, bw=params.get("bw", 0.4), ew=params.get("ew", 0.6))
    return None


def model_exists(dataset: str, model: str, **params) -> bool:
    if model == "tfidf":
        return (model_path(dataset, "tfidf") / "model.joblib").exists()
    if model == "bm25":
        return (model_path(dataset, "bm25", k1=params.get("k1", 1.5), b=params.get("b", 0.75)) / "model.joblib").exists()
    if model == "embedding":
        base = model_path(dataset, "embedding")
        return (base / "embeddings.npz").exists() and (base / "meta.json").exists()
    if model == "hybrid_serial":
        return (model_path(dataset, "hybrid_serial") / "model.pkl").exists()
    if model == "hybrid_parallel":
        return (model_path(dataset, "hybrid_parallel", bw=params.get("bw", 0.4), ew=params.get("ew", 0.6)) / "model.pkl").exists()
    return False


def build_and_save_all_models(
    dataset: str,
    doc_ids: List[str],
    texts: List[str],
    models: Optional[List[tuple]] = None,
) -> Dict[str, float]:
    """Fit all default models, save to disk. Returns elapsed seconds per model."""
    import time

    from services.retrieval import (
        Bm25Retriever,
        EmbeddingRetriever,
        HybridParallelRetriever,
        HybridSerialRetriever,
        TfidfRetriever,
    )

    if models is None:
        models = DEFAULT_MODELS

    timings: Dict[str, float] = {}
    builders = {
        "tfidf": lambda: TfidfRetriever(),
        "bm25": lambda p: Bm25Retriever(k1=p.get("k1", 1.5), b=p.get("b", 0.75)),
        "embedding": lambda: EmbeddingRetriever(),
        "hybrid_serial": lambda: HybridSerialRetriever(),
        "hybrid_parallel": lambda p: HybridParallelRetriever(
            bm25_weight=p.get("bw", 0.4), emb_weight=p.get("ew", 0.6)
        ),
    }

    for model_name, params in models:
        print(f"  Building model: {model_name} ...")
        t0 = time.time()
        if model_name == "bm25":
            r = builders["bm25"](params)
        elif model_name == "hybrid_parallel":
            r = builders["hybrid_parallel"](params)
        else:
            r = builders[model_name]()
        r.fit(doc_ids, texts)
        save_retriever(dataset, model_name, r, **params)
        timings[model_name] = round(time.time() - t0, 2)

    return timings
