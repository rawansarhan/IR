"""BM25 (Okapi) Retriever."""

from __future__ import annotations
#
from typing import List, Tuple
#
import numpy as np

from services.preprocessing import preprocess_text


class Bm25Retriever:
    """
    نموذج BM25 (Best Match 25).

    Matching:  BM25 Okapi Scoring
    ──────────────────────────────
    BM25 يحسب score لكل وثيقة بناءً على:
        - TF (term frequency): كم مرة الكلمة موجودة بالوثيقة
        - IDF (inverse doc frequency): كم الكلمة نادرة بالمجموعة كلها
        - طول الوثيقة: يعاقب الوثائق الطويلة التي تحتوي كلمة بشكل طبيعي

    الصيغة:
        score(D,Q) = Σ IDF(qi) × [ TF(qi,D) × (k1+1) ] / [ TF(qi,D) + k1×(1-b+b×|D|/avgdl) ]

    المعاملات:
        k1 : يتحكم بمدى تشبع تأثير التكرار (قيمة افتراضية 1.5)
             → k1 عالي = تكرار الكلمة أهم
             → k1 منخفض = تكرار الكلمة أقل أهمية
        b  : يتحكم بتطبيع طول الوثيقة (قيمة افتراضية 0.75)
             → b=1 = تطبيع كامل لطول الوثيقة
             → b=0 = بدون تطبيع

    Ranking: ترتيب تنازلي حسب BM25 score
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1   # معامل تشبع تأثير التكرار
        self.b = b     # معامل تطبيع طول الوثيقة
        self.doc_ids: List[str] = []
        self._bm25 = None

    def fit(self, doc_ids: List[str], texts: List[str]) -> None:
        """بناء BM25 index: tokenize كل الوثائق وبناء الـ index."""
        from rank_bm25 import BM25Okapi
        self.doc_ids = doc_ids
        # tokenize: كل وثيقة تصبح قائمة كلمات بعد preprocessing
        tokenized = [preprocess_text(text).split() for text in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
#للبحث عن الوثائق المتشابهة بواسطة  BM25
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Matching & Ranking:
        1. tokenize الاستعلام بنفس طريقة الوثائق
        2. احسب BM25 score لكل وثيقة
        3. رتّب تنازلياً وارجع أعلى top_k
        """
        if self._bm25 is None:
            raise RuntimeError("Retriever is not fitted yet.")

        # تمثيل الاستعلام كقائمة tokens
        tokenized_query = preprocess_text(query).split()

        # BM25 scoring: يرجع array من الدرجات لكل وثيقة
        scores = self._bm25.get_scores(tokenized_query)

        # Ranking: ترتيب تنازلي
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
