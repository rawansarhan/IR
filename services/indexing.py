"""Inverted Index implementation.

Structure:
    term -> { doc_id: term_frequency }

Supports:
    - Building from corpus
    - Saving/loading to disk (JSON)
    - Boolean lookup
    - TF and DF stats used by retrievers
"""

from __future__ import annotations

import json
import math
from collections import defaultdict\
#لتحديد المسارات المختلفة من البيانات
from pathlib import Path
#لتحديد الانواع المختلفة من البيانات 
from typing import Dict, Iterator, List, Optional, Tuple

from services.preprocessing import preprocess_text


class InvertedIndex:
    """In-memory inverted index with TF and DF statistics."""

    def __init__(self) -> None:
        # term -> {doc_id -> term_frequency} تكرار كل كلمة بكل وثيقة 
        self._index: Dict[str, Dict[str, int]] = defaultdict(dict)
        # doc_id -> total token count (for BM25 length norm) طول كل وثيقة
        self._doc_lengths: Dict[str, int] = {}
        # doc_id -> original position (for ranking stability)
        self._doc_order: List[str] = []
        #عدد الوثائق الكلي
        self._num_docs: int = 0

    # ------------------------------------------------------------------
    # Build :بناء الفهرس
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        
        #  تحويل النص الى جملة منفصلة بواسطة فراغات مثل :
        #  "Social Media is harmful" => ["social", "media", "harmful"]
# هون منقيم كل الكلمات يلي مالا داعي لنحصل على index مرتب واصغر
        tokens = preprocess_text(text).split()
        self._doc_lengths[doc_id] = len(tokens)
        if doc_id not in self._doc_order:
            self._doc_order.append(doc_id)
            self._num_docs += 1
        # تحسب التكرار لكل كلمة في المستند 
        # "social social media" => tf = {social : 2 , media : 1}
        tf: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        # تحسب التكرار لكل كلمة بالمستند
        for token, freq in tf.items():
            self._index[token][doc_id] = freq

    def build(self, doc_ids: List[str], texts: List[str]) -> None:
        print(f"  Building inverted index for {len(doc_ids)} documents ...")
        for doc_id, text in zip(doc_ids, texts):
            self.add_document(doc_id, text)
        print(f"  Index ready: {len(self._index):,} unique terms, {self._num_docs:,} docs.")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
#لحساب العدد الاجمالي للمستندات 
    @property
    def num_docs(self) -> int:
        return self._num_docs
# لحساب الطول المتوسط للمستندات
    @property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)
# لحساب عدد الوثائق يلي فيها الكلمة
    def df(self, term: str) -> int:
        """Document frequency — number of docs containing the term."""
        return len(self._index.get(term, {}))
# لحساب عدد مرات تكرار الكلمة في المستند 
    def tf(self, term: str, doc_id: str) -> int:
        """Term frequency of term in doc."""
        return self._index.get(term, {}).get(doc_id, 0)
# لحساب طول الوثيقة 
    def doc_length(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def postings(self, term: str) -> Dict[str, int]:
        """Return posting list {doc_id: tf} for a term."""
        return self._index.get(term, {})
#لحساب القاموس الكلي
# كل المصطلحات للتصحيح الاملائي (يستخد في الاقتراحات Query Refinement  )
    def vocabulary(self) -> List[str]:
        return list(self._index.keys())

    # ------------------------------------------------------------------
    # TF-IDF scoring from index
    # ------------------------------------------------------------------
    #لحساب النتائج النهائية للبحث 
    def tfidf_scores(self, query_terms: List[str]) -> Dict[str, float]:
        """Compute TF-IDF cosine similarity scores for all matching docs."""
        scores: Dict[str, float] = defaultdict(float)
        N = self._num_docs
#
        for term in query_terms:
            df = self.df(term)
            if df == 0:
                continue
            idf = math.log((N + 1) / (df + 1)) + 1  # sklearn-style smooth IDF
            for doc_id, tf in self.postings(term).items():
                scores[doc_id] += tf * idf
#
        return dict(scores)

    # ------------------------------------------------------------------
    # Boolean retrieval helpers
    # ------------------------------------------------------------------
#لحساب النتائج التهائية للبحث بواسطة المعادلة المنطقية و
    def boolean_and(self, terms: List[str]) -> set[str]:
        if not terms:
            return set()
        result = set(self.postings(terms[0]).keys())
        for term in terms[1:]:
            result &= set(self.postings(term).keys())
        return result

    def boolean_or(self, terms: List[str]) -> set[str]:
        result: set[str] = set()
        for term in terms:
            result |= set(self.postings(term).keys())
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
# JSON لحفظ القهرس في ملف 
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "index": {term: postings for term, postings in self._index.items()},
            "doc_lengths": self._doc_lengths,
            "doc_order": self._doc_order,
            "num_docs": self._num_docs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  Index saved to {path} ({path.stat().st_size / 1024:.1f} KB)")
#لتحميل الفهرس من الملف المحفوظ 
    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._index = defaultdict(dict, data["index"])
        self._doc_lengths = data["doc_lengths"]
        self._doc_order = data["doc_order"]
        self._num_docs = data["num_docs"]
        print(f"  Index loaded from {path}: {len(self._index):,} terms, {self._num_docs:,} docs.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
#لعرض الملخص للفهرس
    def summary(self) -> Dict[str, object]:
        return {
            "num_docs": self._num_docs,
            "num_terms": len(self._index),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "top_10_terms": sorted(
                self._index.keys(),
                key=lambda t: len(self._index[t]),
                reverse=True,
            )[:10],
        }

#
def build_doc_lookup(doc_ids: List[str], texts: List[str]) -> Dict[str, str]:
    return dict(zip(doc_ids, texts))
