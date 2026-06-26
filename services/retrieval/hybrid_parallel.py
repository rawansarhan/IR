"""Hybrid Parallel Retriever — BM25 + Embedding fused with weights."""

from __future__ import annotations

from typing import List, Tuple

from services.retrieval.bm25 import Bm25Retriever
from services.retrieval.embedding import EmbeddingRetriever
from services.retrieval.fusion import fuse_parallel


class HybridParallelRetriever:
    """
    Hybrid Parallel (تفرعي) — نموذجان يعملان بالتوازي.

    Matching & Ranking:
    ───────────────────
    1. BM25 يبحث بشكل مستقل ويرجع pool من النتائج
    2. Embedding يبحث بشكل مستقل ويرجع pool من النتائج
    3. fuse_parallel() تدمج النتائج بـ Weighted Score Fusion
    4. الترتيب النهائي حسب الدرجة المدموجة

    الأوزان الافتراضية:
        BM25 weight = 0.4  (40% من الدرجة النهائية)
        Emb  weight = 0.6  (60% — نعطيه وزن أكبر للمعنى الدلالي)
    """

    def __init__(self, bm25_weight: float = 0.4, emb_weight: float = 0.6) -> None:
        self.bm25_weight = bm25_weight
        self.emb_weight = emb_weight
        self._bm25 = Bm25Retriever()
        self._emb = EmbeddingRetriever()

    def fit(self, doc_ids: List[str], texts: List[str]) -> None:
        print("  [Parallel Hybrid] Fitting BM25 ...")
        self._bm25.fit(doc_ids, texts)
        print("  [Parallel Hybrid] Fitting Embedding ...")
        self._emb.fit(doc_ids, texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        # pool أكبر من top_k لضمان تغطية أفضل عند الدمج
        pool = max(top_k * 5, 50)

        # البحث المتوازي — كل نموذج مستقل
        bm25_results = self._bm25.search(query, top_k=pool)
        emb_results = self._emb.search(query, top_k=pool)

        # دمج النتائج بـ Weighted Fusion
        return fuse_parallel(
            [bm25_results, emb_results],
            weights=[self.bm25_weight, self.emb_weight],
            top_k=top_k,
        )
