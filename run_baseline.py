"""Run retrieval baseline — supports tfidf and bm25."""

from __future__ import annotations

import argparse

from core.dataset_loader import build_corpus, load_qrels, load_queries
from services.evaluation import evaluate_run
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

MODELS = {
    "tfidf":            TfidfRetriever,
    "bm25":             Bm25Retriever,
    "embedding":        EmbeddingRetriever,
    "hybrid_serial":    HybridSerialRetriever,
    "hybrid_parallel":  HybridParallelRetriever,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="touche")
    parser.add_argument("--model", choices=MODELS.keys(), default="tfidf")
    parser.add_argument("--limit", type=int, default=None, help="Use fewer docs on weak machines")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dataset_id = DATASETS[args.dataset]
    print(f"\nDataset : {args.dataset} ({dataset_id})")
    print(f"Model   : {args.model}")
    print(f"Limit   : {args.limit or 'all'}")
    print("-" * 40)

    print("Building corpus ...")
    doc_ids, texts = build_corpus(dataset_id, limit=args.limit)
    queries = load_queries(dataset_id)
    qrels = load_qrels(dataset_id)

    print(f"Docs: {len(doc_ids)} | Queries: {len(queries)}")

    retriever = MODELS[args.model]()
    print(f"Fitting {args.model} ...")
    retriever.fit(doc_ids, texts)

    print("Searching ...")
    run = {}
    for query_id, query_text in queries.items():
        results = retriever.search(query_text, top_k=args.top_k)
        run[query_id] = [doc_id for doc_id, _ in results]

    metrics = evaluate_run(run, qrels, k=args.top_k)
    print("\nEvaluation Results:")
    print(f"  MAP              : {metrics['map']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")
    print(f"  Precision@10     : {metrics['precision_at_10']:.4f}")
    print(f"  nDCG             : {metrics['ndcg']:.4f}")
    print(f"  Queries evaluated: {int(metrics['num_queries'])}")


if __name__ == "__main__":
    main()
