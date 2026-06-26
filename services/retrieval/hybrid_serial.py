"""Hybrid Serial Retriever — BM25 candidates then Embedding rerank."""

from __future__ import annotations

from typing import List, Tuple

from services.retrieval.bm25 import Bm25Retriever
from services.retrieval.embedding import EmbeddingRetriever


class HybridSerialRetriever:
    """
    Hybrid Serial (تسلسلي) — مرحلتان متتاليتان.

    Matching & Ranking:
    ───────────────────
    المرحلة 1 (Candidate Retrieval):
        → BM25 يسترجع أفضل candidate_k وثيقة (مثلاً 100)
        → سريع وفعّال في التصفية الأولية

    المرحلة 2 (Reranking):
        → Embedding يعيد ترتيب الـ candidates فقط (مش كل الوثائق)
        → أدق لأنه يفهم المعنى، وأسرع لأنه يعمل على عدد أقل

    الميزة: يجمع سرعة BM25 مع دقة Embeddings
    """

    def __init__(self, candidate_k: int = 100) -> None:
        self.candidate_k = candidate_k  # عدد الـ candidates من BM25
        self._bm25 = Bm25Retriever()
        self._emb = EmbeddingRetriever()

    def fit(self, doc_ids: List[str], texts: List[str]) -> None:
        self._doc_text_map = dict(zip(doc_ids, texts))
        print("  [Serial Hybrid] Fitting BM25 ...")
        self._bm25.fit(doc_ids, texts)
        print("  [Serial Hybrid] Fitting Embedding ...")
        self._emb.fit(doc_ids, texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # المرحلة 1: BM25 يجيب أفضل candidate_k وثيقة
        candidates = self._bm25.search(query, top_k=self.candidate_k)
        candidate_ids = [doc_id for doc_id, _ in candidates]

        # المرحلة 2: Embedding يعيد ترتيب الـ candidates بـ Cosine Similarity
        candidate_texts = [self._doc_text_map.get(doc_id, "") for doc_id in candidate_ids]
        temp_emb = EmbeddingRetriever(self._emb.model_name)
        temp_emb._model = self._emb._get_model()
        temp_emb.fit(candidate_ids, candidate_texts)
        return temp_emb.search(query, top_k=top_k)
