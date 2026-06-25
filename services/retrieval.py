"""Retrieval Service — Query Matching & Ranking.

كل نموذج يعمل خطوتين:
    1. Matching  — مطابقة الاستعلام مع الوثائق
    2. Ranking   — ترتيب النتائج من الأعلى score للأدنى

طريقة المطابقة حسب كل نموذج:
    TF-IDF   → Cosine Similarity بين vector الاستعلام و matrix الوثائق
    BM25     → BM25 Okapi scoring function (تأخذ بعين الاعتبار TF + IDF + طول الوثيقة)
    Embedding→ Cosine Similarity بين embedding الاستعلام و embeddings الوثائق
    Hybrid   → دمج النموذجين (Serial أو Parallel)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing import preprocess_text

#لحساب النتائج النهائية للبحث بواسطة المعادلة المنطقية 
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

        # Cosine Similarity بين الاستعلام وكل الوثائق
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # Ranking: ترتيب تنازلي — أعلى score أولًا
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]


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


def fuse_parallel(
    runs: List[List[Tuple[str, float]]],
    weights: List[float] | None = None,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Fusion Method للـ Hybrid Parallel.

    طريقة الدمج: Weighted Score Fusion مع Normalization
    ────────────────────────────────────────────────────
    لكل نموذج:
        1. normalize الدرجات (÷ max_score) لتصبح بين 0 و 1
        2. اضرب بالوزن المخصص للنموذج
        3. اجمع الدرجات لكل وثيقة من كل النماذج

    مثال:
        BM25 score لـ doc1 = 14.0 → normalized = 1.0 → weighted = 0.4
        Emb  score لـ doc1 = 0.85 → normalized = 0.9 → weighted = 0.54
        Final score doc1 = 0.4 + 0.54 = 0.94
    """
    if not runs:
        return []
    if weights is None:
        weights = [1.0 / len(runs)] * len(runs)

    fused_scores: Dict[str, float] = {}
    for run, weight in zip(runs, weights):
        if not run:
            continue
        # Normalization: قسّم كل score على أعلى score بهذا النموذج
        max_score = max(s for _, s in run) or 1.0
        for doc_id, score in run:
            normalized = score / max_score
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight * normalized

    # Ranking: رتّب تنازلياً حسب الدرجة المدموجة
    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


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
        emb_results  = self._emb.search(query, top_k=pool)

        # دمج النتائج بـ Weighted Fusion
        return fuse_parallel(
            [bm25_results, emb_results],
            weights=[self.bm25_weight, self.emb_weight],
            top_k=top_k,
        )
