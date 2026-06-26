"""حفظ/تحميل الفهرس المعكوس (Inverted Index) كملف JSON مضغوط."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from services.indexing import InvertedIndex
from services.persistence.paths import index_path


def save_index(dataset: str, index: InvertedIndex) -> Path:
    path = index_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "index": {term: postings for term, postings in index._index.items()},
        "doc_lengths": index._doc_lengths,
        "doc_order": index._doc_order,
        "num_docs": index._num_docs,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def load_index(dataset: str) -> Optional[InvertedIndex]:
    path = index_path(dataset)
    if not path.exists():
        return None
    idx = InvertedIndex()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    idx._index = defaultdict(dict, data["index"])
    idx._doc_lengths = data["doc_lengths"]
    idx._doc_order = data["doc_order"]
    idx._num_docs = data["num_docs"]
    return idx


def index_exists(dataset: str) -> bool:
    return index_path(dataset).exists()
