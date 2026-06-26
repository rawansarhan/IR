"""Documents Clustering — group documents by topic using KMeans.

الميزة الإضافية: Documents Clustering
──────────────────────────────────────
نستخدم الـ embeddings الموجودة (Dense vectors من all-MiniLM-L6-v2) ونطبّق
KMeans لتجميع الوثائق المتشابهة دلاليًا في عناقيد (clusters).

لكل عنقود:
    - عدد الوثائق
    - أهم الكلمات المميّزة (top terms) عبر TF-IDF على وثائق العنقود
    - عيّنة من معرّفات الوثائق

تُحفظ النتيجة على القرص لإعادة الاستخدام بدون إعادة حساب.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from services.preprocessing import preprocess_text

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CLUSTER_DIR = DATA_ROOT / "clusters"


def _top_terms_per_cluster(
    texts: List[str],
    labels: np.ndarray,
    n_clusters: int,
    top_n: int = 8,
) -> Dict[int, List[str]]:
    """Compute the most distinctive terms for each cluster via TF-IDF centroid."""
    processed = [preprocess_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(processed)
    terms = np.array(vectorizer.get_feature_names_out())

    top_terms: Dict[int, List[str]] = {}
    for c in range(n_clusters):
        mask = labels == c
        if not mask.any():
            top_terms[c] = []
            continue
        # mean TF-IDF vector for the cluster
        centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[::-1][:top_n]
        top_terms[c] = [terms[i] for i in top_idx if centroid[i] > 0]
    return top_terms


def cluster_documents(
    dataset: str,
    doc_ids: List[str],
    texts: List[str],
    embeddings: np.ndarray,
    n_clusters: int = 8,
    sample_size: int = 5,
) -> dict:
    """Run KMeans on document embeddings and summarize each cluster."""
    embeddings = embeddings.astype("float32")
    n_clusters = max(2, min(n_clusters, len(doc_ids)))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    top_terms = _top_terms_per_cluster(texts, labels, n_clusters)

    clusters = []
    for c in range(n_clusters):
        member_idx = np.where(labels == c)[0]
        sample_ids = [doc_ids[i] for i in member_idx[:sample_size]]
        clusters.append(
            {
                "cluster_id": int(c),
                "size": int(len(member_idx)),
                "top_terms": top_terms.get(c, []),
                "label": " · ".join(top_terms.get(c, [])[:3]) or f"Cluster {c}",
                "sample_doc_ids": sample_ids,
            }
        )

    clusters.sort(key=lambda x: x["size"], reverse=True)

    result = {
        "dataset": dataset,
        "n_clusters": n_clusters,
        "total_docs": len(doc_ids),
        "inertia": round(float(kmeans.inertia_), 2),
        "clusters": clusters,
    }
    return result


def save_clusters(dataset: str, result: dict) -> Path:
    base = CLUSTER_DIR
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{dataset}_clusters.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return out


def load_clusters(dataset: str) -> Optional[dict]:
    out = CLUSTER_DIR / f"{dataset}_clusters.json"
    if not out.exists():
        return None
    with open(out, encoding="utf-8") as f:
        return json.load(f)
