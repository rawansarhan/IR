"""Text preprocessing pipeline.

Steps applied (in order):
1. Lowercase
2. Remove punctuation & digits
3. Tokenize
4. Remove stopwords
5. Stemming  (PorterStemmer)
6. Lemmatization (WordNetLemmatizer)
7. Rejoin tokens
"""

from __future__ import annotations

import re
import string

import nltk

# Download required NLTK data once
for _pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if _pkg.startswith("punkt") else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    stem: bool = True,
    lemmatize: bool = True,
) -> str:
    if not text:
        return ""

    # 1. Lowercase
    if lowercase:
        text = text.lower()

    # 2. Remove punctuation and digits
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenize
    tokens = word_tokenize(text)

    # 5. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    # 6. Lemmatize first (needs real words)
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # 7. Stem after lemmatize
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)
