"""Load datasets from ir_datasets without keeping everything in memory."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import ir_datasets


def load_ir_dataset(dataset_id: str):
    return ir_datasets.load(dataset_id)


def iter_documents(dataset_id: str, limit: Optional[int] = None) -> Iterator[Tuple[str, str]]:
    ds = load_ir_dataset(dataset_id)
    for index, doc in enumerate(ds.docs_iter()):
        if limit is not None and index >= limit:
            break
        text = doc.text or ""
        title = getattr(doc, "title", "") or ""
        full_text = f"{title} {text}".strip()
        yield doc.doc_id, full_text


def load_queries(dataset_id: str) -> Dict[str, str]:
    ds = load_ir_dataset(dataset_id)
    return {query.query_id: query.text for query in ds.queries_iter()}


def load_qrels(dataset_id: str) -> Dict[str, Dict[str, int]]:
    ds = load_ir_dataset(dataset_id)
    qrels: Dict[str, Dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels


def get_dataset_stats(dataset_id: str) -> Dict[str, int]:
    ds = load_ir_dataset(dataset_id)
    return {
        "docs": ds.docs_count(),
        "queries": ds.queries_count(),
        "qrels": ds.qrels_count(),
    }


def build_corpus(dataset_id: str, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    texts: List[str] = []
    for doc_id, text in iter_documents(dataset_id, limit=limit):
        doc_ids.append(doc_id)
        texts.append(text)
    return doc_ids, texts
