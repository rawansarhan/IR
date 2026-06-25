"""FastAPI — IR Project Service Gateway.

Endpoints:
    GET  /health
    GET  /datasets
    POST /load        — load dataset + SQLite + index + pre-build all models
    POST /search      — run a query (models loaded from disk, not trained on query)
    GET  /evaluate    — run evaluation on loaded dataset
    POST /query-steps — show preprocessing steps for a query
    POST /refine       — query refinement (spell correct, suggestions, PRF)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.dataset_loader import build_corpus, load_qrels, load_queries
from services.document_store import DocumentStore
from services.evaluation import evaluate_run
from services.indexing import InvertedIndex
from services.persistence import (
    build_and_save_all_models,
    index_exists,
    load_index,
    load_retriever,
    model_exists,
    retriever_cache_key,
    save_index,
)
from services.query_processing import log_query_transformation
from services.query_refinement import refine_query

app = FastAPI(title="IR Project API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ──────────────────────────────────────────────────
DATASETS = {
    "touche": "beir/webis-touche2020",
    "quora": "beir/quora/test",
}

_doc_store = DocumentStore()
_corpus: Dict[str, tuple[list, list]] = {}   # key -> (doc_ids, texts) — texts cached after load
_queries: Dict[str, dict] = {}
_qrels: Dict[str, dict] = {}
_retrievers: Dict[str, object] = {}          # "touche:bm25:..." -> retriever
_indexes: Dict[str, InvertedIndex] = {}


# ── Schemas ──────────────────────────────────────────────────────────
class LoadRequest(BaseModel):
    dataset: str                  # touche | quora
    limit: Optional[int] = None   # None = full dataset
    rebuild: bool = False         # force rebuild even if cached on disk


class SearchRequest(BaseModel):
    dataset: str
    query: str
    model: str = "bm25"          # tfidf | bm25 | embedding | hybrid_serial | hybrid_parallel
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_weight: float = 0.4
    emb_weight: float = 0.6


class QueryStepsRequest(BaseModel):
    query: str


class RefineRequest(BaseModel):
    dataset: str
    query: str
    model: str = "bm25"
    enable_spell_correct: bool = True
    enable_suggestions: bool = True
    enable_prf: bool = True


class EvaluateAllRequest(BaseModel):
    dataset: str
    top_k: int = 10
    mode: str = "baseline"       # baseline | enhanced


# ── Helpers ──────────────────────────────────────────────────────────
def _api_retriever_key(dataset: str, model: str, **params) -> str:
    return f"{dataset}:{retriever_cache_key(model, **params)}"


def _get_retriever(
    dataset: str,
    model: str,
    doc_ids: List[str],
    texts: List[str],
    **params,
) -> object:
    """Load retriever from memory or disk. Never fit on first search."""
    key = _api_retriever_key(
        dataset, model,
        k1=params.get("k1", 1.5), b=params.get("b", 0.75),
        bw=params.get("bw", 0.4), ew=params.get("ew", 0.6),
    )

    if key in _retrievers:
        return _retrievers[key]

    load_params = {
        "k1": params.get("k1", 1.5),
        "b": params.get("b", 0.75),
        "bw": params.get("bw", 0.4),
        "ew": params.get("ew", 0.6),
    }

    if model_exists(dataset, model, **load_params):
        retriever = load_retriever(dataset, model, **load_params)
        if retriever is not None:
            _retrievers[key] = retriever
            return retriever

    raise HTTPException(
        400,
        f"Model '{model}' not built for '{dataset}'. "
        f"Call POST /load first (models are pre-built on load, not on search).",
    )


def _ensure_index(dataset: str, doc_ids: List[str], texts: List[str], rebuild: bool = False) -> InvertedIndex:
    if dataset in _indexes and not rebuild:
        return _indexes[dataset]

    if not rebuild and index_exists(dataset):
        idx = load_index(dataset)
        if idx is not None:
            _indexes[dataset] = idx
            return idx

    idx = InvertedIndex()
    idx.build(doc_ids, texts)
    save_index(dataset, idx)
    _indexes[dataset] = idx
    return idx


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    loaded = {
        ds: _doc_store.count(ds)
        for ds in DATASETS
        if _doc_store.is_loaded(ds)
    }
    return {"status": "ok", "loaded_datasets": list(_corpus.keys()), "db_docs": loaded}


@app.get("/datasets")
def list_datasets():
    return {
        "available": list(DATASETS.keys()),
        "loaded": list(_corpus.keys()),
        "db_loaded": [ds for ds in DATASETS if _doc_store.is_loaded(ds)],
    }


@app.post("/load")
def load_dataset(req: LoadRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(404, f"Unknown dataset: {req.dataset}")

    ds_id = DATASETS[req.dataset]
    t0 = time.time()
    steps: Dict[str, object] = {}

    # 1) Load corpus from ir_datasets
    t1 = time.time()
    doc_ids, texts = build_corpus(ds_id, limit=req.limit)
    steps["corpus_sec"] = round(time.time() - t1, 2)

    # 2) Persist to SQLite
    t2 = time.time()
    stored = _doc_store.store_documents(req.dataset, doc_ids, texts)
    steps["sqlite_sec"] = round(time.time() - t2, 2)
    steps["docs_stored"] = stored

    _corpus[req.dataset] = (doc_ids, texts)
    _queries[req.dataset] = load_queries(ds_id)
    _qrels[req.dataset] = load_qrels(ds_id)

    # 3) Build inverted index + save compressed
    t3 = time.time()
    idx = _ensure_index(req.dataset, doc_ids, texts, rebuild=req.rebuild)
    steps["index_sec"] = round(time.time() - t3, 2)
    steps["index_terms"] = len(idx._index)

    # 4) Pre-build ALL models + save to disk (not on first search)
    all_cached = (
        not req.rebuild
        and all(model_exists(req.dataset, m, **p) for m, p in [
            ("tfidf", {}),
            ("bm25", {"k1": 1.5, "b": 0.75}),
            ("embedding", {}),
            ("hybrid_serial", {}),
            ("hybrid_parallel", {"bw": 0.4, "ew": 0.6}),
        ])
    )

    if all_cached and not req.rebuild:
        steps["models_sec"] = 0
        steps["models"] = "loaded_from_cache"
        # Warm memory cache
        for model, params in [
            ("tfidf", {}),
            ("bm25", {"k1": 1.5, "b": 0.75}),
            ("embedding", {}),
            ("hybrid_serial", {}),
            ("hybrid_parallel", {"bw": 0.4, "ew": 0.6}),
        ]:
            key = _api_retriever_key(req.dataset, model, **params)
            if key not in _retrievers:
                r = load_retriever(req.dataset, model, **params)
                if r:
                    _retrievers[key] = r
    else:
        _retrievers.clear()
        t4 = time.time()
        model_timings = build_and_save_all_models(req.dataset, doc_ids, texts)
        steps["models_sec"] = round(time.time() - t4, 2)
        steps["model_timings"] = model_timings
        # Load into memory
        for model, params in [
            ("tfidf", {}),
            ("bm25", {"k1": 1.5, "b": 0.75}),
            ("embedding", {}),
            ("hybrid_serial", {}),
            ("hybrid_parallel", {"bw": 0.4, "ew": 0.6}),
        ]:
            key = _api_retriever_key(req.dataset, model, **params)
            _retrievers[key] = load_retriever(req.dataset, model, **params)

    elapsed = round(time.time() - t0, 2)
    return {
        "dataset": req.dataset,
        "docs_loaded": len(doc_ids),
        "queries": len(_queries[req.dataset]),
        "elapsed_sec": elapsed,
        "storage": "sqlite",
        "models_prebuilt": True,
        "steps": steps,
    }


@app.post("/search")
def search(req: SearchRequest):
    if req.dataset not in _corpus and not _doc_store.is_loaded(req.dataset):
        raise HTTPException(400, f"Dataset '{req.dataset}' not loaded. Call /load first.")

    doc_ids, texts = _corpus.get(req.dataset, ([], []))
    if not doc_ids and _doc_store.is_loaded(req.dataset):
        doc_ids = _doc_store.doc_ids(req.dataset)
        texts = []  # not needed if models on disk

    retriever = _get_retriever(
        req.dataset, req.model, doc_ids, texts,
        k1=req.bm25_k1, b=req.bm25_b,
        bw=req.bm25_weight, ew=req.emb_weight,
    )

    t0 = time.time()
    results = retriever.search(req.query, top_k=req.top_k)
    elapsed = round(time.time() - t0, 4)

    return {
        "query": req.query,
        "model": req.model,
        "dataset": req.dataset,
        "elapsed_sec": elapsed,
        "results": [
            {
                "rank": i + 1,
                "doc_id": doc_id,
                "score": round(score, 4),
                "snippet": (_doc_store.get_document(req.dataset, doc_id) or "")[:200],
            }
            for i, (doc_id, score) in enumerate(results)
        ],
    }


@app.get("/evaluate")
def evaluate(dataset: str, model: str = "bm25", top_k: int = 10):
    if dataset not in _corpus and not _doc_store.is_loaded(dataset):
        raise HTTPException(400, f"Dataset '{dataset}' not loaded. Call /load first.")

    doc_ids, texts = _corpus.get(dataset, ([], []))
    queries = _queries.get(dataset, {})
    qrels = _qrels.get(dataset, {})

    retriever = _get_retriever(dataset, model, doc_ids, texts)
    run = {
        qid: [doc_id for doc_id, _ in retriever.search(qtext, top_k=top_k)]
        for qid, qtext in queries.items()
    }

    metrics = evaluate_run(run, qrels, k=top_k)
    return {"dataset": dataset, "model": model, "top_k": top_k, "metrics": metrics}


@app.get("/evaluate/all")
def evaluate_all(dataset: str, top_k: int = 10):
    """Evaluate all 5 models and return comparison table."""
    if dataset not in _corpus and not _doc_store.is_loaded(dataset):
        raise HTTPException(400, f"Dataset '{dataset}' not loaded. Call /load first.")

    models = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
    results = []
    for model in models:
        try:
            data = evaluate(dataset=dataset, model=model, top_k=top_k)
            results.append({"model": model, **data["metrics"]})
        except HTTPException as e:
            results.append({"model": model, "error": e.detail})

    return {"dataset": dataset, "top_k": top_k, "results": results}


@app.post("/query-steps")
def query_steps(req: QueryStepsRequest):
    steps = log_query_transformation(req.query)
    return {"original": req.query, "steps": steps}


@app.post("/refine")
def refine_query_endpoint(req: RefineRequest):
    if req.dataset not in _corpus and not _doc_store.is_loaded(req.dataset):
        raise HTTPException(400, f"Dataset '{req.dataset}' not loaded. Call /load first.")

    doc_ids, texts = _corpus.get(req.dataset, ([], []))
    vocabulary: List[str] = []

    if req.dataset in _indexes:
        vocabulary = _indexes[req.dataset].vocabulary()
    elif index_exists(req.dataset):
        idx = load_index(req.dataset)
        if idx:
            _indexes[req.dataset] = idx
            vocabulary = idx.vocabulary()

    top_doc_texts: List[str] = []

    if req.enable_prf:
        retriever = _get_retriever(req.dataset, req.model, doc_ids, texts)
        initial = retriever.search(req.query, top_k=3)
        top_doc_texts = [
            _doc_store.get_document(req.dataset, doc_id) or ""
            for doc_id, _ in initial
        ]
        top_doc_texts = [t for t in top_doc_texts if t]

    result = refine_query(
        req.query,
        vocabulary=vocabulary if vocabulary else None,
        top_doc_texts=top_doc_texts if top_doc_texts else None,
        enable_spell_correct=req.enable_spell_correct and bool(vocabulary),
        enable_suggestions=req.enable_suggestions and bool(vocabulary),
        enable_prf=req.enable_prf and bool(top_doc_texts),
    )

    return {
        "dataset": req.dataset,
        "model_used_for_prf": req.model if req.enable_prf else None,
        "index_available": bool(vocabulary),
        **result,
    }


@app.post("/index/{dataset}")
def build_index(dataset: str):
    if dataset not in _corpus and not _doc_store.is_loaded(dataset):
        raise HTTPException(400, f"Dataset '{dataset}' not loaded. Call /load first.")

    doc_ids, texts = _corpus.get(dataset, ([], []))
    if not texts:
        doc_ids = _doc_store.doc_ids(dataset)
        texts = [_doc_store.get_document(dataset, d) or "" for d in doc_ids]

    t0 = time.time()
    idx = _ensure_index(dataset, doc_ids, texts, rebuild=True)
    elapsed = round(time.time() - t0, 2)
    summary = idx.summary()

    return {
        "dataset": dataset,
        "status": "ready",
        "saved_to_disk": True,
        "elapsed_sec": elapsed,
        **summary,
    }


@app.get("/index/{dataset}/stats")
def index_stats(dataset: str):
    if dataset not in _indexes:
        if index_exists(dataset):
            _indexes[dataset] = load_index(dataset)
        else:
            raise HTTPException(400, f"Index for '{dataset}' not built yet. Call POST /load or /index/{dataset}.")

    summary = _indexes[dataset].summary()
    return {"dataset": dataset, "status": "ready", "on_disk": index_exists(dataset), **summary}


@app.get("/index/{dataset}/term/{term}")
def term_lookup(dataset: str, term: str):
    if dataset not in _indexes:
        if index_exists(dataset):
            _indexes[dataset] = load_index(dataset)
        else:
            raise HTTPException(400, f"Index for '{dataset}' not built yet.")

    idx = _indexes[dataset]
    postings = idx.postings(term)
    return {
        "term": term,
        "document_frequency": idx.df(term),
        "postings_sample": dict(list(postings.items())[:10]),
    }


@app.get("/docs/{dataset}")
def list_docs(dataset: str, page: int = 1, page_size: int = 20):
    if not _doc_store.is_loaded(dataset):
        raise HTTPException(400, f"Dataset '{dataset}' not loaded. Call /load first.")

    total, docs = _doc_store.list_page(dataset, page=page, page_size=page_size)
    return {
        "dataset": dataset,
        "total_docs": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "docs": docs,
        "source": "sqlite",
    }


@app.get("/docs/{dataset}/{doc_id:path}")
def get_doc(dataset: str, doc_id: str):
    if not _doc_store.is_loaded(dataset):
        raise HTTPException(400, f"Dataset '{dataset}' not loaded. Call /load first.")

    text = _doc_store.get_document(dataset, doc_id)
    if text is None:
        raise HTTPException(404, f"Document '{doc_id}' not found.")

    return {
        "dataset": dataset,
        "doc_id": doc_id,
        "text": text,
        "length": len(text.split()),
        "source": "sqlite",
    }
