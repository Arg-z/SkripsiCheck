"""TF-IDF lexical similarity and exact-text detection."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def is_exact_match(text_a: str, text_b: str) -> bool:
    """Compare text after case and whitespace normalization."""

    return bool(text_a.strip()) and normalized_tokens(text_a) == normalized_tokens(text_b)


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Calculate pairwise TF-IDF cosine similarity in the range 0..1."""

    if not text_a.strip() or not text_b.strip():
        return 0.0
    if is_exact_match(text_a, text_b):
        return 1.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(
            [text_a, text_b]
        )
    except ValueError:
        return 0.0
    return float(np.clip(cosine_similarity(matrix[0], matrix[1])[0, 0], 0.0, 1.0))


def tfidf_against_sources(query: str, sources: Sequence[str]) -> list[float]:
    """Compare one query against many source passages in a shared vector space."""

    if not query.strip() or not sources:
        return [0.0] * len(sources)
    corpus = [query, *sources]
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(corpus)
    except ValueError:
        return [0.0] * len(sources)
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    return [float(np.clip(score, 0.0, 1.0)) for score in scores]

