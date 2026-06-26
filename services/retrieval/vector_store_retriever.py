"""Vector Store Retriever — semantic search backed by a FAISS index."""

from __future__ import annotations

from typing import List, Tuple


class VectorStoreRetriever:
    """
    نموذج بحث دلالي مبني على Vector Store (FAISS).

    الفرق عن EmbeddingRetriever العادي:
    ───────────────────────────────────
    - EmbeddingRetriever: يبحث بضرب مصفوفة كامل (O(N) لكل استعلام) → بطيء على البيانات الكبيرة
    - VectorStoreRetriever: يخزّن الـ embeddings في فهرس FAISS متخصص للبحث السريع

    Matching: Cosine Similarity عبر FAISS IndexFlatIP (vectors مُطبّعة)
    Ranking:  ترتيب تنازلي حسب درجة التشابه التي يرجّعها FAISS
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.doc_ids: List[str] = []
        self._model = None
        self._store = None  # FaissVectorStore

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, doc_ids: List[str], texts: List[str], batch_size: int = 64) -> None:
        """تحويل الوثائق لـ embeddings ثم بناء فهرس FAISS."""
        from services.vector_store import FaissVectorStore

        model = self._get_model()
        self.doc_ids = doc_ids
        print(f"  [VectorStore] Encoding {len(texts)} documents ...")
        # يحول كل الوثائق الى embeddings 
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print("  [VectorStore] Building FAISS index ...")
        # ينشئ قاعدة بيانات FAISS.
        self._store = FaissVectorStore()
        #هنا يبدأ بناء الـ Index. 
        self._store.build(doc_ids, embeddings)

    def fit_from_embeddings(self, doc_ids: List[str], embeddings) -> None:
        """بناء الفهرس من embeddings محسوبة مسبقًا (إعادة استخدام)."""
        from services.vector_store import FaissVectorStore

        self.doc_ids = doc_ids
        self._store = FaissVectorStore()
        self._store.build(doc_ids, embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._store is None:
            raise RuntimeError("Vector store retriever is not fitted yet.")
        model = self._get_model()
        #تحويل الاستعلام الى embedding
        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # FAISS يقوم بالبحث داخل الـ Index مباشرة.=>Top K
        return self._store.search(query_emb, top_k=top_k)
