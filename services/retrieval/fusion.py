"""Fusion methods for combining results of parallel retrievers."""

from __future__ import annotations

from typing import Dict, List, Tuple


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
    # قائمة نتائج كل النماذج 
    if not runs:
        return []
    if weights is None:
        #لتسوية الدرجات الى قيم  بن 1 و0 
        weights = [1.0 / len(runs)] * len(runs)
# قائمة الدرجات المدموجة لكل وثيقة
    fused_scores: Dict[str, float] = {}
    for run, weight in zip(runs, weights):
        if not run:
            continue
        # Normalization: قسّم كل score على أعلى score بهذا النموذج
        # مناخد اعلى درجة بكل نموذج ويقسم عليها الدرجات الاخرى
        max_score = max(s for _, s in run) or 1.0
        for doc_id, score in run:
            normalized = score / max_score
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight * normalized

    # Ranking: رتّب تنازلياً حسب الدرجة المدموجة
    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]
