"""Retrieval Service — Query Matching & Ranking.

هذا الـ package يجمّع كل نماذج التمثيل والاسترجاع، كل نموذج في ملف مستقل:

    tfidf.py                  → TfidfRetriever        (VSM + Cosine Similarity)
    bm25.py                   → Bm25Retriever         (BM25 Okapi)
    embedding.py              → EmbeddingRetriever    (Dense / BERT)
    fusion.py                 → fuse_parallel         (Weighted Score Fusion)
    hybrid_serial.py          → HybridSerialRetriever (BM25 ثم Embedding rerank)
    hybrid_parallel.py        → HybridParallelRetriever (BM25 + Embedding + Fusion)
    vector_store_retriever.py → VectorStoreRetriever  (FAISS)

يُعاد تصدير كل الأصناف هنا للحفاظ على التوافق مع:
    from services.retrieval import TfidfRetriever, Bm25Retriever, ...
"""

from __future__ import annotations

from services.retrieval.bm25 import Bm25Retriever
from services.retrieval.embedding import EmbeddingRetriever
from services.retrieval.fusion import fuse_parallel
from services.retrieval.hybrid_parallel import HybridParallelRetriever
from services.retrieval.hybrid_serial import HybridSerialRetriever
from services.retrieval.tfidf import TfidfRetriever
from services.retrieval.vector_store_retriever import VectorStoreRetriever

__all__ = [
    "TfidfRetriever",
    "Bm25Retriever",
    "EmbeddingRetriever",
    "fuse_parallel",
    "HybridSerialRetriever",
    "HybridParallelRetriever",
    "VectorStoreRetriever",
]
