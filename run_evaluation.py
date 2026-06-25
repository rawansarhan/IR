"""Full evaluation — all models, baseline vs enhanced (with Query Refinement).

يحسب MAP, Recall, P@10, nDCG لكل نموذج ولكل dataset.
يقارن:
    - baseline  : بحث مباشر بدون ميزات إضافية
    - enhanced  : مع Query Refinement (spell correct + PRF)

النتائج تُحفظ في results/evaluation_report.csv للتقرير.

Usage:
    python run_evaluation.py --dataset touche --limit 5000
    python run_evaluation.py --dataset touche --models bm25,embedding
    python run_evaluation.py --dataset touche --mode both
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Type

from core.dataset_loader import build_corpus, load_qrels, load_queries
from services.evaluation import evaluate_run
from services.indexing import InvertedIndex
from services.query_refinement import refine_query
from services.retrieval import (
    Bm25Retriever,
    EmbeddingRetriever,
    HybridParallelRetriever,
    HybridSerialRetriever,
    TfidfRetriever,
)

DATASETS = {
    "touche": "beir/webis-touche2020",
    "quora": "beir/quora/test",
}

MODELS: Dict[str, Type] = {
    "tfidf": TfidfRetriever,
    "bm25": Bm25Retriever,
    "embedding": EmbeddingRetriever,
    "hybrid_serial": HybridSerialRetriever,
    "hybrid_parallel": HybridParallelRetriever,
}

# نماذج سريعة للتقييم على جهاز ضعيف (بدون embedding)
FAST_MODELS = ["tfidf", "bm25"]


def run_search(
    retriever,
    queries: Dict[str, str],
    top_k: int,
    refined_queries: Dict[str, str] | None = None,
) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for qid, qtext in queries.items():
        q = refined_queries.get(qid, qtext) if refined_queries else qtext
        results = retriever.search(q, top_k=top_k)
        run[qid] = [doc_id for doc_id, _ in results]
    return run


def build_refined_queries(
    queries: Dict[str, str],
    vocabulary: List[str],
    retriever,
    top_k: int = 10,
) -> Dict[str, str]:
    """Refine كل استعلام باستخدام PRF + spell correction."""
    refined: Dict[str, str] = {}
    for qid, qtext in queries.items():
        initial = retriever.search(qtext, top_k=3)
        # نحتاج نصوص الوثائق — نستخدم doc_ids فقط كـ placeholder
        # PRF يحتاج نصوص؛ نمرّر قائمة فارغة إذا ما عندنا map
        top_texts = [doc_id for doc_id, _ in initial]  # fallback
        result = refine_query(
            qtext,
            vocabulary=vocabulary,
            top_doc_texts=top_texts if top_texts else None,
            enable_spell_correct=bool(vocabulary),
            enable_suggestions=False,
            enable_prf=False,  # PRF يحتاج نصوص حقيقية
        )
        refined[qid] = str(result["refined"])
    return refined


def build_refined_queries_full(
    queries: Dict[str, str],
    vocabulary: List[str],
    retriever,
    doc_map: Dict[str, str],
) -> Dict[str, str]:
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


def evaluate_model(
    model_name: str,
    doc_ids: List[str],
    texts: List[str],
    queries: Dict[str, str],
    qrels: Dict[str, dict],
    top_k: int,
    mode: str,
    vocabulary: List[str],
) -> dict:
    log(f"\n  [{mode.upper()}] {model_name} ...")
    t0 = time.time()

    retriever = MODELS[model_name]()
    retriever.fit(doc_ids, texts)

    doc_map = dict(zip(doc_ids, texts))
    refined = None
    if mode == "enhanced":
        refined = build_refined_queries_full(
            queries, vocabulary, retriever, doc_map
        )

    run = run_search(retriever, queries, top_k, refined)
    metrics = evaluate_run(run, qrels, k=top_k)
    elapsed = round(time.time() - t0, 2)

    return {**metrics, "elapsed_sec": elapsed, "model": model_name, "mode": mode}


def print_table(rows: List[dict]) -> None:
    header = f"{'Dataset':<8} {'Mode':<10} {'Model':<18} {'MAP':>8} {'Recall':>8} {'P@10':>8} {'nDCG':>8} {'Time':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:<8} {r['mode']:<10} {r['model']:<18} "
            f"{r['map']:>8.4f} {r['recall']:>8.4f} "
            f"{r['precision_at_10']:>8.4f} {r['ndcg']:>8.4f} "
            f"{r['elapsed_sec']:>7.1f}s"
        )
    print("=" * len(header))


def log(msg: str) -> None:
    print(msg, flush=True)


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
    log(f"\nSaved -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IR System Full Evaluation")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["both"], default="touche")
    parser.add_argument("--models", default="fast",
                        help="Comma-separated model names, or 'fast' (tfidf+bm25) or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["baseline", "enhanced", "both"], default="both",
                        help="baseline=without features, enhanced=with query refinement")
    parser.add_argument("--output", default="results/evaluation_report.csv")
    args = parser.parse_args()

    if args.models == "all":
        model_list = list(MODELS.keys())
    elif args.models == "fast":
        model_list = FAST_MODELS
    else:
        model_list = [m.strip() for m in args.models.split(",")]

    datasets = list(DATASETS.keys()) if args.dataset == "both" else [args.dataset]
    modes = ["baseline", "enhanced"] if args.mode == "both" else [args.mode]

    all_rows: List[dict] = []

    for ds_key in datasets:
        ds_id = DATASETS[ds_key]
        log(f"\n{'='*50}")
        log(f"Dataset: {ds_key} ({ds_id})")
        log(f"{'='*50}")

        log("Loading corpus (full dataset, no limit) ...")
        doc_ids, texts = build_corpus(ds_id, limit=args.limit)
        queries = load_queries(ds_id)
        qrels = load_qrels(ds_id)
        log(f"Docs: {len(doc_ids):,} | Queries: {len(queries)} | Qrels: {sum(len(v) for v in qrels.values())}")

        log("Building inverted index for refinement ...")
        idx = InvertedIndex()
        idx.build(doc_ids, texts)
        vocabulary = idx.vocabulary()

        for mode in modes:
            for model_name in model_list:
                try:
                    row = evaluate_model(
                        model_name, doc_ids, texts, queries, qrels,
                        args.top_k, mode, vocabulary,
                    )
                    row["dataset"] = ds_key
                    all_rows.append(row)
                    save_csv(all_rows, Path(args.output))
                    log(
                        f"  Done {ds_key}/{mode}/{model_name}: "
                        f"MAP={row['map']:.4f} Recall={row['recall']:.4f} "
                        f"P@10={row['precision_at_10']:.4f} nDCG={row['ndcg']:.4f}"
                    )
                except Exception as e:
                    log(f"  ERROR on {model_name}: {e}")

    print_table(all_rows)
    save_csv(all_rows, Path(args.output))

    log("\nSummary for Report:")
    baseline_rows = [r for r in all_rows if r["mode"] == "baseline"]
    enhanced_rows = [r for r in all_rows if r["mode"] == "enhanced"]
    if baseline_rows:
        best = max(baseline_rows, key=lambda r: r["map"])
        print(f"  Best baseline model : {best['model']} on {best['dataset']} (MAP={best['map']:.4f})")
    if enhanced_rows:
        best_e = max(enhanced_rows, key=lambda r: r["map"])
        print(f"  Best enhanced model : {best_e['model']} on {best_e['dataset']} (MAP={best_e['map']:.4f})")
    if baseline_rows and enhanced_rows:
        for model in model_list:
            b = next((r for r in baseline_rows if r["model"] == model), None)
            e = next((r for r in enhanced_rows if r["model"] == model), None)
            if b and e:
                diff = e["map"] - b["map"]
                sign = "+" if diff >= 0 else ""
                print(f"  {model}: MAP {sign}{diff:.4f} after refinement")


if __name__ == "__main__":
    main()
