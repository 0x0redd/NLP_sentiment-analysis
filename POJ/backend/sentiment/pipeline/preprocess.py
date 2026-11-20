from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for resource in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

NEGATIONS = {
    "no", "not", "nor", "neither", "never", "n't",
    "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
}
STOP_ALL = set(stopwords.words("english"))
STOP_KEEP_NEG = STOP_ALL - NEGATIONS
_stemmer = PorterStemmer()


def _tokenize_clean(text: str) -> list[str]:
    text = re.sub(r"<br\s*/?>", " ", str(text).lower())
    return [t for t in word_tokenize(text) if t.isalpha()]


def preprocess_article(text: str) -> str:
    tokens = [t for t in _tokenize_clean(text) if t not in STOP_ALL]
    return " ".join(_stemmer.stem(t) for t in tokens)


def preprocess_negation(text: str) -> str:
    tokens = [t for t in _tokenize_clean(text) if t not in STOP_KEEP_NEG]
    return " ".join(_stemmer.stem(t) for t in tokens)
