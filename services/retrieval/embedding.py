"""Embedding (Dense) Retriever using Sentence-Transformers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class EmbeddingRetriever:
    """
    نموذج Embedding (Dense Retrieval) باستخدام Sentence Transformers.

    Matching:  Cosine Similarity بين Dense Embeddings
    ──────────────────────────────────────────────────
    - كل وثيقة تتحول لـ dense vector (768 أو 384 بُعد) يمثّل معناها
    - الاستعلام يتحول لـ dense vector بنفس الطريقة
    - نحسب cosine similarity بين vector الاستعلام وكل وثيقة
    - الميزة: يفهم المعنى الدلالي، مش بس التطابق الحرفي
      مثال: "ban" و"prohibit" بيطلعوا قريبين بالـ embedding space

    النموذج المستخدم: all-MiniLM-L6-v2 (BERT-based, 384 dim)

    Ranking: ترتيب تنازلي حسب cosine similarity
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.doc_ids: List[str] = []
        self.doc_embeddings = None  # numpy array: (num_docs × embedding_dim)
        self._model = None

    def _get_model(self):
        """تحميل النموذج (مرة وحدة فقط - lazy loading)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, doc_ids: List[str], texts: List[str], batch_size: int = 64) -> None:
        """تحويل كل الوثائق لـ embeddings وحفظها."""
        model = self._get_model()
        self.doc_ids = doc_ids
        print(f"  Encoding {len(texts)} documents (model: {self.model_name}) ...")
        self.doc_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # normalize للـ cosine similarity
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Matching & Ranking:
        1. حوّل الاستعلام لـ embedding vector
        2. احسب cosine similarity = dot product (بعد normalize)
        3. رتّب تنازلياً وارجع أعلى top_k
        """
        if self.doc_embeddings is None:
            raise RuntimeError("Retriever is not fitted yet.")
        model = self._get_model()

        # تمثيل الاستعلام كـ embedding
        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Cosine Similarity = dot product بعد normalize
        # query_emb shape: (1, dim) × doc_embeddings.T shape: (dim, num_docs)
        scores = (query_emb @ self.doc_embeddings.T).flatten()

        # Ranking: ترتيب تنازلي
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
