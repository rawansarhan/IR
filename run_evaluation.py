"""Full evaluation — all models, before vs after extra features.

يحسب MAP, Recall, P@10, nDCG لكل نموذج ولكل dataset.

يعيد استخدام النماذج المحفوظة على القرص (persistence) إن وُجدت، وإلا يبنيها
ويحفظها — هكذا يتم الترميز (encoding) مرة واحدة فقط ويُعاد استخدامه.

أوضاع المقارنة:
    - baseline : النماذج الأساسية (قبل الميزات الإضافية)
    - enhanced : مع Query Refinement (spell correct + PRF)
    - extra    : Vector Store (FAISS) — الميزة الإضافية (نفس جودة Embedding بسرعة أعلى)

النتائج تُحفظ في results/evaluation_report.csv للتقرير.

Usage:
    python run_evaluation.py --dataset touche --models all --mode baseline
    python run_evaluation.py --dataset both --models all --mode all
    python run_evaluation.py --dataset touche --models bm25,vector_store
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

from core.dataset_loader import build_corpus, load_qrels, load_queries
from services.evaluation import evaluate_run
from services.indexing import InvertedIndex
from services.persistence import (
    build_and_save_all_models,
    load_index,
    load_retriever,
    model_exists,
    save_index,
)
from services.query_refinement import refine_query
from services.retrieval import VectorStoreRetriever
from services.vector_store import FaissVectorStore

DATASETS = {
    "touche": "beir/webis-touche2020",
    "quora": "beir/quora/test",
}

BASE_MODELS = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
EXTRA_MODELS = ["vector_store"]
ALL_MODELS = BASE_MODELS + EXTRA_MODELS

MODEL_PARAMS = {
    "tfidf": {},
    "bm25": {"k1": 1.5, "b": 0.75},
    "embedding": {},
    "hybrid_serial": {},
    "hybrid_parallel": {"bw": 0.4, "ew": 0.6},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_models(dataset: str, doc_ids: List[str], texts: List[str]) -> None:
    """Build + save all base models once if not already on disk."""
    missing = [m for m, p in MODEL_PARAMS.items() if not model_exists(dataset, m, **p)]
    if missing:
        log(f"  Building/saving missing models: {missing} (one-time encoding) ...")
        build_and_save_all_models(dataset, doc_ids, texts)
    else:
        log("  All base models already cached on disk.")


def ensure_vector_store(dataset: str) -> bool:
    if FaissVectorStore.exists(dataset):
        return True
    emb = load_retriever(dataset, "embedding")
    if emb is None or emb.doc_embeddings is None:
        return False
    store = FaissVectorStore()
    store.build(emb.doc_ids, emb.doc_embeddings)
    store.save(dataset)
    return True


def get_retriever(dataset: str, model: str):
    if model == "vector_store":
        store = FaissVectorStore.load(dataset)
        r = VectorStoreRetriever()
        r.doc_ids = store.doc_ids
        r._store = store
        return r
    return load_retriever(dataset, model, **MODEL_PARAMS.get(model, {}))


def build_refined_queries(queries, vocabulary, retriever, doc_map) -> Dict[str, str]:
    refined: Dict[str, str] = {}
    for qid, qtext in queries.items():
        initial = retriever.search(qtext, top_k=3)
        top_texts = [doc_map.get(doc_id, "") for doc_id, _ in initial]
        result = refine_query(
            qtext,
            vocabulary=vocabulary,
            top_doc_texts=[t for t in top_texts if t],
            enable_spell_correct=bool(vocabulary),
            enable_suggestions=False,
            enable_prf=bool(top_texts),
        )
        refined[qid] = str(result["refined"])
    return refined


def evaluate_model(dataset, model, queries, qrels, top_k, mode, doc_map, vocabulary) -> dict:
    log(f"\n  [{mode.upper()}] {model} ...")
    t0 = time.time()
    retriever = get_retriever(dataset, model)
    if retriever is None:
        raise RuntimeError(f"Retriever '{model}' not available for '{dataset}'")

    refined = None
    if mode == "enhanced":
        refined = build_refined_queries(queries, vocabulary, retriever, doc_map)

    run: Dict[str, List[str]] = {}
    for qid, qtext in queries.items():
        q = refined.get(qid, qtext) if refined else qtext
        run[qid] = [doc_id for doc_id, _ in retriever.search(q, top_k=top_k)]

    metrics = evaluate_run(run, qrels, k=top_k)
    elapsed = round(time.time() - t0, 2)
    return {**metrics, "elapsed_sec": elapsed, "model": model, "mode": mode}


def save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "mode", "model", "map", "recall",
        "precision_at_10", "ndcg", "num_queries", "elapsed_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    log(f"  Saved -> {path}")


def print_table(rows: List[dict]) -> None:
    header = f"{'Dataset':<8} {'Mode':<10} {'Model':<18} {'MAP':>8} {'Recall':>8} {'P@10':>8} {'nDCG':>8} {'Time':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:<8} {r['mode']:<10} {r['model']:<18} "
            f"{r['map']:>8.4f} {r['recall']:>8.4f} "
            f"{r['precision_at_10']:>8.4f} {r['ndcg']:>8.4f} {r['elapsed_sec']:>8.1f}s"
        )
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="IR System Full Evaluation")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["both"], default="touche")
    parser.add_argument("--models", default="all",
                        help="'all', 'base', 'extra', or comma-separated names")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["baseline", "enhanced", "all"], default="baseline")
    parser.add_argument("--output", default="results/evaluation_report.csv")
    args = parser.parse_args()

    if args.models == "all":
        model_list = ALL_MODELS
    elif args.models == "base":
        model_list = BASE_MODELS
    elif args.models == "extra":
        model_list = EXTRA_MODELS
    else:
        model_list = [m.strip() for m in args.models.split(",")]

    datasets = list(DATASETS.keys()) if args.dataset == "both" else [args.dataset]
    modes = ["baseline", "enhanced"] if args.mode == "all" else [args.mode]

    all_rows: List[dict] = []
    out_path = Path(args.output)

    for ds_key in datasets:
        ds_id = DATASETS[ds_key]
        log(f"\n{'='*60}\nDataset: {ds_key} ({ds_id})\n{'='*60}")

        log("Loading corpus ...")
        doc_ids, texts = build_corpus(ds_id, limit=args.limit)
        queries = load_queries(ds_id)
        qrels = load_qrels(ds_id)
        doc_map = dict(zip(doc_ids, texts))
        log(f"Docs: {len(doc_ids):,} | Queries: {len(queries)} | Qrels: {sum(len(v) for v in qrels.values())}")

        # inverted index (for refinement vocabulary) — reuse if saved
        idx = load_index(ds_key)
        if idx is None:
            log("Building inverted index ...")
            idx = InvertedIndex()
            idx.build(doc_ids, texts)
            save_index(ds_key, idx)
        vocabulary = idx.vocabulary()

        # one-time model build + save
        base_needed = [m for m in model_list if m in BASE_MODELS]
        if base_needed:
            ensure_models(ds_key, doc_ids, texts)
        if "vector_store" in model_list:
            if not ensure_vector_store(ds_key):
                log("  WARNING: cannot build vector_store (embedding missing)")

        for mode in modes:
            for model in model_list:
                try:
                    row = evaluate_model(
                        ds_key, model, queries, qrels, args.top_k, mode, doc_map, vocabulary
                    )
                    row["dataset"] = ds_key
                    all_rows.append(row)
                    save_csv(all_rows, out_path)
                    log(
                        f"  Done {ds_key}/{mode}/{model}: "
                        f"MAP={row['map']:.4f} R={row['recall']:.4f} "
                        f"P@10={row['precision_at_10']:.4f} nDCG={row['ndcg']:.4f} "
                        f"({row['elapsed_sec']}s)"
                    )
                except Exception as e:
                    log(f"  ERROR on {model}: {e}")

    print_table(all_rows)
    save_csv(all_rows, out_path)

    # before/after extra features summary
    log("\n--- Before vs After Extra Features (MAP) ---")
    for ds_key in datasets:
        emb = next((r for r in all_rows if r["dataset"] == ds_key and r["model"] == "embedding" and r["mode"] == "baseline"), None)
        vs = next((r for r in all_rows if r["dataset"] == ds_key and r["model"] == "vector_store" and r["mode"] == "baseline"), None)
        if emb and vs:
            speedup = emb["elapsed_sec"] / vs["elapsed_sec"] if vs["elapsed_sec"] else 0
            log(f"  {ds_key}: embedding MAP={emb['map']:.4f} ({emb['elapsed_sec']}s) "
                f"-> vector_store MAP={vs['map']:.4f} ({vs['elapsed_sec']}s) | speedup x{speedup:.1f}")


if __name__ == "__main__":
    main()
