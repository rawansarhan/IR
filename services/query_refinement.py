"""Query Refinement Service.

يوفّر:
    1. تطبيع الاستعلام (Normalization)
    2. اقتراح مصطلحات من الفهرس (Query Suggestions)
    3. تصحيح إملائي بسيط مقابل مفردات الفهرس (Spelling Correction)
    4. توسيع الاستعلام بـ Pseudo Relevance Feedback (PRF)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from difflib import get_close_matches
from typing import Dict, List, Optional

from services.preprocessing import preprocess_text
from services.query_processing import process_query


def normalize_query(query: str) -> str:
    """إزالة المسافات الزائدة وتوحيد الشكل."""
    return " ".join(query.split()).strip()


def _query_tokens(query: str) -> List[str]:
    return process_query(query).split()


def suggest_from_vocabulary(
    query: str,
    vocabulary: List[str],
    max_suggestions: int = 5,
) -> List[str]:
    """
    اقتراح مصطلحات من مفردات الفهرس غير موجودة بالاستعلام.
    يفضّل المصطلحات الشائعة (أول القائمة = الأكثر تكرارًا).
    """
    query_lower = set(_query_tokens(query))
    suggestions = []
    for term in vocabulary:
        if term not in query_lower and len(term) > 2:
            suggestions.append(term)
        if len(suggestions) >= max_suggestions:
            break
    return suggestions


def spell_correct_query(
    query: str,
    vocabulary: List[str],
    cutoff: float = 0.75,
) -> tuple[str, List[dict]]:
    """
    تصحيح إملائي بسيط: لكل token نبحث عن أقرب كلمة في الفهرس.
    يرجع الاستعلام المصحّح + قائمة التصحيحات.
    """
    tokens = query.split()
    vocab_set = set(vocabulary)
    corrected_tokens = []
    corrections: List[dict] = []

    for token in tokens:
        clean = token.lower().strip(string.punctuation)
        if not clean:
            corrected_tokens.append(token)
            continue

        if clean in vocab_set:
            corrected_tokens.append(token)
            continue

        matches = get_close_matches(clean, vocabulary, n=1, cutoff=cutoff)
        if matches:
            corrected_tokens.append(matches[0])
            corrections.append({"original": token, "corrected": matches[0]})
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens), corrections


def extract_top_terms_from_docs(texts: List[str], top_n: int = 5) -> List[str]:
    """استخراج أكثر المصطلحات تكرارًا من نصوص الوثائق (لـ PRF)."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(process_query(text).split())
    return [term for term, _ in counter.most_common(top_n)]


def pseudo_relevance_expand(
    query: str,
    top_doc_texts: List[str],
    max_terms: int = 3,
) -> str:
    """
    Pseudo Relevance Feedback:
    نفترض أن أفضل الوثائق المسترجعة ذات صلة،
    ونضيف أكثر مصطلحاتها تكرارًا للاستعلام.
    """
    expansion_terms = extract_top_terms_from_docs(top_doc_texts, top_n=max_terms + 5)
    query_tokens = set(_query_tokens(query))
    selected = [t for t in expansion_terms if t not in query_tokens][:max_terms]
    if not selected:
        return query
    return f"{query} {' '.join(selected)}"


def refine_query(
    query: str,
    vocabulary: Optional[List[str]] = None,
    top_doc_texts: Optional[List[str]] = None,
    enable_spell_correct: bool = True,
    enable_suggestions: bool = True,
    enable_prf: bool = True,
) -> Dict[str, object]:
    """
    خط أنابيب كامل لتحسين الاستعلام.

    يرجع:
        original, normalized, spell_corrected, corrections,
        suggestions, prf_expanded, refined (الاستعلام النهائي)
    """
    original = query
    normalized = normalize_query(query)

    result: Dict[str, object] = {
        "original": original,
        "normalized": normalized,
        "spell_corrected": normalized,
        "corrections": [],
        "suggestions": [],
        "prf_expanded": normalized,
        "refined": normalized,
        "steps_applied": [],
    }

    current = normalized

    # 1. تصحيح إملائي
    if enable_spell_correct and vocabulary:
        corrected, corrections = spell_correct_query(current, vocabulary)
        if corrections:
            result["corrections"] = corrections
            result["steps_applied"].append("spell_correction")
        current = corrected
        result["spell_corrected"] = current

    # 2. اقتراحات
    if enable_suggestions and vocabulary:
        result["suggestions"] = suggest_from_vocabulary(current, vocabulary)

    # 3. PRF
    if enable_prf and top_doc_texts:
        expanded = pseudo_relevance_expand(current, top_doc_texts)
        if expanded != current:
            result["steps_applied"].append("pseudo_relevance_feedback")
        current = expanded
        result["prf_expanded"] = current

    result["refined"] = current
    return result
