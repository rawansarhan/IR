"""Interactive demo — type a query and see top-10 results."""

from __future__ import annotations

from core.dataset_loader import build_corpus, load_queries, load_qrels
from services.retrieval import Bm25Retriever, TfidfRetriever

DATASETS = {
    "touche": "beir/webis-touche2020",
    "quora":  "beir/quora/test",
}

MODELS = {
    "tfidf": TfidfRetriever,
    "bm25":  Bm25Retriever,
}


def main() -> None:
    print("=== IR Demo ===\n")

    ds_choice = input("Dataset (touche / quora) [touche]: ").strip() or "touche"
    model_choice = input("Model   (tfidf / bm25)  [bm25]:   ").strip() or "bm25"
    limit = int(input("Docs limit (e.g. 5000, 0 = all) [5000]: ").strip() or "5000")

    dataset_id = DATASETS.get(ds_choice, DATASETS["touche"])
    limit = limit if limit > 0 else None

    print(f"\nLoading corpus ({limit or 'all'} docs) ...")
    doc_ids, texts = build_corpus(dataset_id, limit=limit)
    doc_text_map = dict(zip(doc_ids, texts))

    # Load qrels to show relevance labels
    qrels = load_qrels(dataset_id)
    all_qrels_doc_ids: set[str] = set()
    for rel_map in qrels.values():
        all_qrels_doc_ids.update(rel_map.keys())

    print(f"Fitting {model_choice} on {len(doc_ids)} docs ...")
    retriever = MODELS[model_choice]()
    retriever.fit(doc_ids, texts)

    print("\nReady! Type your query (or 'quit' to exit).\n")

    while True:
        query = input("Query > ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = retriever.search(query, top_k=10)

        print(f"\nTop-10 results for: \"{query}\"\n")
        print(f"{'#':<4} {'Score':>7}  {'Doc ID':<20}  Snippet")
        print("-" * 80)

        for rank, (doc_id, score) in enumerate(results, start=1):
            snippet = doc_text_map.get(doc_id, "")[:120].replace("\n", " ")
            print(f"{rank:<4} {score:>7.4f}  {doc_id:<20}  {snippet}")

        print()


if __name__ == "__main__":
    main()
