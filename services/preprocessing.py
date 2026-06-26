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
# استخدمنا ال set لان البحث فيها اسرع من ال list
# مجموعة كلمات الللغة الانكليزية
_STOPWORDS = set(stopwords.words("english"))
# تستخدم لتخفسض الكلمات الى اساسية  مثل running => run
_STEMMER = PorterStemmer()
# اداة الاشتقاق المعجمي 
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

    # 2. Remove punctuation and digits (ازالة الترقيم والارقام)
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 3. Normalize whitespace (تسوية المسافات الزائدة)
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenize (تقسيم النص لكلمات)
    tokens = word_tokenize(text)

    # 5. Remove stopwords (ازالة الكلمات الشائعة بلا معنى)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    # 6. Lemmatize first (needs real words) (تجويل الكلمات الى الجذر)
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # 7. Stem after lemmatize (قص نهايات الجذر )
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
  # اعادة تجميع القائمة لنص واحد => (["water", "ban"] → "water ban".)
    return " ".join(tokens)
