"""Save/load indexes and retrieval models to compressed files on disk."""

from __future__ import annotations

import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from services.indexing import InvertedIndex
from services.retrieval import (
    Bm25Retriever,
    EmbeddingRetriever,
    HybridParallelRetriever,
    HybridSerialRetriever,
    TfidfRetriever,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_ROOT / "indexes"
MODEL_DIR = DATA_ROOT / "models"

# Default models built on /load
DEFAULT_MODELS = [
    ("tfidf", {}),
    ("bm25", {"k1": 1.5, "b": 0.75}),
    ("embedding", {}),
    ("hybrid_serial", {}),
    ("hybrid_parallel", {"bw": 0.4, "ew": 0.6}),
]


def index_path(dataset: str) -> Path:
    return INDEX_DIR / f"{dataset}_index.json.gz"


def model_path(dataset: str, model: str, **params) -> Path:
    base = MODEL_DIR / dataset
    if model == "bm25":
        name = f"bm25_k1={params.get('k1', 1.5)}_b={params.get('b', 0.75)}"
    elif model == "hybrid_parallel":
        name = f"hybrid_parallel_bw={params.get('bw', 0.4)}_ew={params.get('ew', 0.6)}"
    else:
        name = model
    return base / name


def retriever_cache_key(model: str, **params) -> str:
    if model == "bm25":
        return f"{model}:k1={params.get('k1', 1.5)}:b={params.get('b', 0.75)}"
    if model == "hybrid_parallel":
        return f"{model}:bw={params.get('bw', 0.4)}:ew={params.get('ew', 0.6)}"
    return model


# ── Index persistence ────────────────────────────────────────────────

def save_index(dataset: str, index: InvertedIndex) -> Path:
    path = index_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "index": {term: postings for term, postings in index._index.items()},
        "doc_lengths": index._doc_lengths,
        "doc_order": index._doc_order,
        "num_docs": index._num_docs,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def load_index(dataset: str) -> Optional[InvertedIndex]:
    path = index_path(dataset)
    if not path.exists():
        return None
    idx = InvertedIndex()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    idx._index = defaultdict(dict, data["index"])
    idx._doc_lengths = data["doc_lengths"]
    idx._doc_order = data["doc_order"]
    idx._num_docs = data["num_docs"]
    return idx


def index_exists(dataset: str) -> bool:
    return index_path(dataset).exists()


# ── Model persistence ────────────────────────────────────────────────

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
    from services.retrieval import (
        Bm25Retriever,
        EmbeddingRetriever,
        HybridParallelRetriever,
        HybridSerialRetriever,
        TfidfRetriever,
    )
    import time

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
