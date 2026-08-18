"""Word-shingle overlap for likely copied phrases."""

from __future__ import annotations

from app.core.lexical_similarity import normalized_tokens


def word_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    """Build unique, case-insensitive word n-grams."""

    if size < 1:
        raise ValueError("Shingle size must be positive.")
    tokens = normalized_tokens(text)
    if not tokens:
        return set()
    effective_size = min(size, len(tokens))
    return {
        tuple(tokens[index : index + effective_size])
        for index in range(len(tokens) - effective_size + 1)
    }


def _shingles_from_tokens(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def ngram_overlap(query: str, source: str, size: int = 5) -> float:
    """Return the fraction of query shingles also found in the source."""

    if size < 1:
        raise ValueError("Shingle size must be positive.")
    query_tokens = normalized_tokens(query)
    source_tokens = normalized_tokens(source)
    if not query_tokens or not source_tokens:
        return 0.0
    effective_size = min(size, len(query_tokens), len(source_tokens))
    query_shingles = _shingles_from_tokens(query_tokens, effective_size)
    source_shingles = _shingles_from_tokens(source_tokens, effective_size)
    return len(query_shingles & source_shingles) / len(query_shingles)
