from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List

# إعداد الـ logger — بيطلع كل شي بالتيرمنال مع الوقت واسم الملف
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluation")


def _relevant_docs(qrels_for_query: Dict[str, int]) -> set[str]:
    """
    استخرج الوثائق ذات الصلة من الـ qrels.
    نعتبر الوثيقة صحيحة إذا كانت درجة صلتها > 0
    """
    relevant = {doc_id for doc_id, rel in qrels_for_query.items() if rel > 0}
    logger.debug(f"Relevant docs count: {len(relevant)}")
    return relevant


def precision_at_k(retrieved: List[str], relevant: set[str], k: int) -> float:
    """
    احسب Precision@K — من أول K وثيقة، كم واحدة صح؟
    """
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    score = hits / k
    logger.debug(f"Precision@{k}: hits={hits}/{k} → {score:.4f}")
    return score


def average_precision(retrieved: List[str], relevant: set[str]) -> float:
    """
    احسب Average Precision لاستعلام واحد.
    يكافئ النظام اللي يرجع الوثائق الصح بالمراكز الأولى.
    """
    if not relevant:
        logger.debug("AP: no relevant docs → 0.0")
        return 0.0

    hits = 0
    score = 0.0
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            p = hits / index
            score += p
            logger.debug(f"  AP hit at rank {index}: doc={doc_id}, precision={p:.4f}, cumulative={score:.4f}")

    ap = score / len(relevant)
    logger.debug(f"AP final: {score:.4f} / {len(relevant)} = {ap:.4f}")
    return ap


def recall_at_k(retrieved: List[str], relevant: set[str], k: int) -> float:
    """
    احسب Recall@K — من كل الصح الموجود، كم وجد النظام في أول K؟
    """
    if not relevant:
        logger.debug("Recall: no relevant docs → 0.0")
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    score = hits / len(relevant)
    logger.debug(f"Recall@{k}: hits={hits}/{len(relevant)} → {score:.4f}")
    return score


def ndcg_at_k(retrieved: List[str], qrels_for_query: Dict[str, int], k: int) -> float:
    """
    احسب nDCG@K — جودة الترتيب مع مراعاة درجات الصلة.
    وثيقة مهمة بالأول أفضل من وثيقة مهمة بالآخر.
    """
    dcg = 0.0
    logger.debug(f"nDCG@{k} calculation:")
    for index, doc_id in enumerate(retrieved[:k], start=1):
        rel = qrels_for_query.get(doc_id, 0)
        gain = (2**rel - 1) / math.log2(index + 1)
        dcg += gain
        if rel > 0:
            logger.debug(f"  rank={index}, doc={doc_id}, rel={rel}, gain={gain:.4f}, DCG={dcg:.4f}")

    ideal_rels = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = 0.0
    for index, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(index + 1)

    if idcg == 0:
        logger.debug("nDCG: IDCG=0, no relevant docs → 0.0")
        return 0.0

    ndcg = dcg / idcg
    logger.debug(f"nDCG: DCG={dcg:.4f}, IDCG={idcg:.4f} → {ndcg:.4f}")
    return ndcg


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    قيّم أداء نظام الاسترجاع على مجموعة استعلامات كاملة.

    المدخلات:
        run   = { query_id: [doc_id1, doc_id2, ...] }
        qrels = { query_id: { doc_id: relevance } }
        k     = عدد النتائج المأخوذة بالحسبان

    المخرجات:
        MAP, Recall, P@10, nDCG
    """
    logger.info(f"Starting evaluation: {len(run)} queries, k={k}")

    map_scores: List[float] = []
    recall_scores: List[float] = []
    p10_scores: List[float] = []
    ndcg_scores: List[float] = []

    skipped = 0
    for query_id, retrieved in run.items():
        if query_id not in qrels:
            logger.debug(f"Skipping query {query_id} — no qrels found")
            skipped += 1
            continue

        logger.debug(f"── Evaluating query: {query_id} | retrieved={len(retrieved)} docs")

        relevant = _relevant_docs(qrels[query_id])

        ap  = average_precision(retrieved, relevant)
        rec = recall_at_k(retrieved, relevant, k)
        p10 = precision_at_k(retrieved, relevant, k)
        ndcg = ndcg_at_k(retrieved, qrels[query_id], k)

        map_scores.append(ap)
        recall_scores.append(rec)
        p10_scores.append(p10)
        ndcg_scores.append(ndcg)

        logger.debug(f"   → AP={ap:.4f} | Recall={rec:.4f} | P@10={p10:.4f} | nDCG={ndcg:.4f}")

    logger.info(f"Evaluation done: {len(map_scores)} queries evaluated, {skipped} skipped")

    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    results = {
        "map":             mean(map_scores),
        "recall":          mean(recall_scores),
        "precision_at_10": mean(p10_scores),
        "ndcg":            mean(ndcg_scores),
        "num_queries":     len(map_scores),
    }

    logger.info(
        f"Final metrics → MAP={results['map']:.4f} | "
        f"Recall={results['recall']:.4f} | "
        f"P@10={results['precision_at_10']:.4f} | "
        f"nDCG={results['ndcg']:.4f}"
    )

    return results
