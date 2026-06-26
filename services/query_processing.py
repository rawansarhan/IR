"""Query Processing Service.

Applies the same preprocessing pipeline used for documents,
then represents the query using the same method (TF-IDF / BM25 / Embedding).

This guarantees alignment between query and document representations.
"""
#لتحويل البحث الى جملة منفصلة بواسطة فراغات 
from __future__ import annotations
# 
from typing import List, Tuple
#
from services.preprocessing import preprocess_text


def process_query(
    query: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    stem: bool = True,
    lemmatize: bool = True,
) -> str:
    """Apply the same preprocessing pipeline used on documents."""
    #
    return preprocess_text(
        query,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_stopwords=remove_stopwords,
        stem=stem,
        lemmatize=lemmatize,
    )

#لتقسيم الاستعلام الى كلمات 
def tokenize_query(query: str) -> List[str]:
    """Preprocess and tokenize query into terms."""
    return process_query(query).split()

#
def expand_query(query: str, synonyms: dict[str, List[str]] | None = None) -> str:
    """Optional: expand query with synonyms or related terms."""
    if not synonyms:
        return query
    tokens = query.split()
    expanded = list(tokens)
    for token in tokens:
        if token in synonyms:
            expanded.extend(synonyms[token])
    return " ".join(expanded)

#
def log_query_transformation(original: str) -> dict:
    """Return a dict showing step-by-step query transformation for the report/UI."""
    import string, re

    steps = {"original": original}

    lowered = original.lower()
    steps["lowercase"] = lowered
# ازالة التلرقيم والارقام
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation + string.digits))
    steps["remove_punctuation"] = no_punct
# تقسيم النص لكلمات 
    tokens = no_punct.split()
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
        no_stop = [t for t in tokens if t not in sw]
    except Exception:
        no_stop = tokens
    steps["remove_stopwords"] = " ".join(no_stop)
#تحويل الكلمات لجذر
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(t) for t in no_stop]
    except Exception:
        lemmatized = no_stop
    steps["lemmatize"] = " ".join(lemmatized)
#قص نهايات الجذر 
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(t) for t in lemmatized]
    except Exception:
        stemmed = lemmatized
    steps["stem"] = " ".join(stemmed)

    steps["final"] = steps["stem"]
    return steps
