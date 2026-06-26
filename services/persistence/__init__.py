"""حزمة الحفظ والتحميل (Persistence): تحفظ الفهرس والنماذج المُدرّبة على القرص.

تعيد تصدير كل الدوال العامة حتى تبقى الاستيرادات القديمة
`from services.persistence import ...` تعمل بدون أي تغيير.
"""

from services.persistence.bm25 import load_bm25, save_bm25
from services.persistence.dispatch import (
    DEFAULT_MODELS,
    build_and_save_all_models,
    load_retriever,
    model_exists,
    save_retriever,
)
from services.persistence.embedding import load_embedding, save_embedding
from services.persistence.hybrid_parallel import load_hybrid_parallel, save_hybrid_parallel
from services.persistence.hybrid_serial import load_hybrid_serial, save_hybrid_serial
from services.persistence.index_store import index_exists, load_index, save_index
from services.persistence.paths import (
    DATA_ROOT,
    INDEX_DIR,
    MODEL_DIR,
    index_path,
    model_path,
    retriever_cache_key,
)
from services.persistence.tfidf import load_tfidf, save_tfidf

__all__ = [
    "DATA_ROOT",
    "INDEX_DIR",
    "MODEL_DIR",
    "DEFAULT_MODELS",
    "index_path",
    "model_path",
    "retriever_cache_key",
    "save_index",
    "load_index",
    "index_exists",
    "save_tfidf",
    "load_tfidf",
    "save_bm25",
    "load_bm25",
    "save_embedding",
    "load_embedding",
    "save_hybrid_serial",
    "load_hybrid_serial",
    "save_hybrid_parallel",
    "load_hybrid_parallel",
    "save_retriever",
    "load_retriever",
    "model_exists",
    "build_and_save_all_models",
]
