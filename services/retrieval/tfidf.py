"""TF-IDF Retriever (Vector Space Model)."""
 #TfidfVectorizer من scikit-learn لحساب TF وIDF وبناء مصفوفة TF-IDF.
 #cosine_similarity من scikit-learn لحساب التشابه بين الاستعلام والوثائق.
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing import preprocess_text


class TfidfRetriever:
    """
    نموذج TF-IDF مع VSM (Vector Space Model).

    Matching:  Cosine Similarity
    ────────────────────────────
    - كل وثيقة تتمثّل كـ vector من أوزان TF-IDF
    - الاستعلام يتمثّل بنفس الطريقة
    - نحسب cosine similarity بين vector الاستعلام وكل وثيقة
    - cos(θ) = (q · d) / (|q| × |d|)
    - القيمة بين 0 و 1 — كلما اقتربت من 1 كلما كانت الوثيقة أكثر صلة

    Ranking: ترتيب تنازلي حسب درجة التشابه
    """
#scikit-learn استخدمنا هذه المكتبة لحساب ال TF , IDF  , TF-IDF Matrix بشكل تلقائي   
    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = None  # sparse matrix: (num_docs × num_terms)

    def fit(self, doc_ids: List[str], texts: List[str]) -> None:
        """بناء الـ TF-IDF matrix لكل الوثائق."""
        processed = [preprocess_text(text) for text in texts]
        self.doc_ids = doc_ids
        # doc_matrix: كل صف = وثيقة، كل عمود = term، القيمة = TF-IDF weight
        self.doc_matrix = self.vectorizer.fit_transform(processed)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Matching & Ranking:
        1. حوّل الاستعلام لـ TF-IDF vector
        2. احسب cosine similarity مع كل الوثائق
        3. رتّب تنازلياً وارجع أعلى top_k
        """
        if self.doc_matrix is None:
            raise RuntimeError("Retriever is not fitted yet.")

        # تمثيل الاستعلام بنفس طريقة الوثائق
        query_vec = self.vectorizer.transform([preprocess_text(query)])

        # Cosine Similarity بين الاستعلام وكل الوثائق  مكتبة جاهزة من  scikit-learn 
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # Ranking: ترتيب تنازلي — أعلى score أولًا
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
