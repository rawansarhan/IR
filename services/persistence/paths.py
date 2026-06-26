"""مسارات الحفظ على القرص + مفاتيح الكاش (cache keys)."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
INDEX_DIR = DATA_ROOT / "indexes"
MODEL_DIR = DATA_ROOT / "models"


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
