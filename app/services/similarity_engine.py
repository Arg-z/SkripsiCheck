"""Candidate retrieval and transparent final scoring for a paragraph.

The engine deliberately depends on a small ``CandidateRetriever`` protocol
instead of FAISS.  This keeps vector retrieval separate from the inexpensive
lexical and phrase-overlap checks, and lets another index implementation be
introduced without changing the scoring pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.config import SETTINGS
from app.core.lexical_similarity import tfidf_similarity
from app.core.ngram_similarity import ngram_overlap
from app.core.scorer import combine_scores


@runtime_checkable
class SemanticCandidate(Protocol):
    """Minimum shape returned by a semantic candidate search."""

    chunk_id: str
    source_file: str
    text: str
    semantic_score: float


@runtime_checkable
class CandidateRetriever(Protocol):
    """Retrieval boundary used by the final similarity engine."""

    def search(self, query: str, top_k: int) -> Sequence[SemanticCandidate]:
        """Return the most semantically relevant source chunks."""


class SourceIndexerSearch(Protocol):
    """Shape of the PHASE 3 source indexer without importing it eagerly."""

    def search_similar_chunks(
        self, query: str, top_k: int | None = None
    ) -> Sequence[SemanticCandidate]:
        """Search indexed chunks using the indexer's public API."""


@dataclass(slots=True)
class SourceIndexerRetriever:
    """Adapt ``SourceIndexer.search_similar_chunks`` to ``CandidateRetriever``."""

    indexer: SourceIndexerSearch

    def search(self, query: str, top_k: int) -> Sequence[SemanticCandidate]:
        return self.indexer.search_similar_chunks(query, top_k=top_k)


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    """Normalized source metadata consumed by the scoring stage."""

    chunk_id: str
    source_file: str
    text: str
    semantic_score: float
    source_path: str | None = None
    word_count: int = 0
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ScoredMatch:
    """All signals and metadata needed by a future report renderer."""

    paragraph: str
    chunk_id: str
    source_file: str
    matched_text: str
    lexical_similarity: float
    semantic_similarity: float
    ngram_overlap: float
    final_score: float
    risk: str
    reason: str
    source_path: str | None = None
    word_count: int = 0
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ParagraphAnalysis:
    """Result of retrieving and scoring candidates for one paragraph."""

    paragraph: str
    candidates_retrieved: int
    matches: tuple[ScoredMatch, ...]


CandidateInput = SemanticCandidate | RetrievedCandidate | Mapping[str, Any]


def _candidate_value(candidate: CandidateInput, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _normalized_candidate(candidate: CandidateInput) -> RetrievedCandidate:
    chunk_id = str(_candidate_value(candidate, "chunk_id", "")).strip()
    source_file = str(_candidate_value(candidate, "source_file", "")).strip()
    text = str(_candidate_value(candidate, "text", "")).strip()

    if not chunk_id:
        raise ValueError("Semantic candidate is missing chunk_id.")
    if not source_file:
        raise ValueError("Semantic candidate is missing source_file.")

    raw_score = float(_candidate_value(candidate, "semantic_score", 0.0))
    if not math.isfinite(raw_score):
        raise ValueError("Semantic candidate score must be finite.")
    # Normalized cosine/IP values can differ from their mathematical limits by
    # a few ulps. Negative cosine means no positive similarity for reporting.
    semantic_score = min(1.0, max(0.0, raw_score))

    source_path_value = _candidate_value(candidate, "source_path")
    source_path = str(source_path_value) if source_path_value is not None else None
    page_value = _candidate_value(candidate, "page")
    page = int(page_value) if page_value is not None else None
    word_count_value = _candidate_value(candidate, "word_count")
    word_count = (
        int(word_count_value) if word_count_value is not None else len(text.split())
    )

    return RetrievedCandidate(
        chunk_id=chunk_id,
        source_file=source_file,
        text=text,
        semantic_score=semantic_score,
        source_path=source_path,
        word_count=word_count,
        page=page,
    )


def _deduplicate_candidates(
    candidates: Sequence[CandidateInput],
) -> list[RetrievedCandidate]:
    """Keep the best semantic hit for each source-file/chunk pair."""

    unique: dict[tuple[str, str], RetrievedCandidate] = {}
    for raw_candidate in candidates:
        candidate = _normalized_candidate(raw_candidate)
        if not candidate.text:
            continue
        key = (candidate.source_file.casefold(), candidate.chunk_id)
        previous = unique.get(key)
        if previous is None or candidate.semantic_score > previous.semantic_score:
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (
            -candidate.semantic_score,
            candidate.source_file.casefold(),
            candidate.chunk_id,
        ),
    )


def retrieve_candidates(
    query: str,
    retriever: CandidateRetriever,
    *,
    top_k: int | None = None,
) -> list[RetrievedCandidate]:
    """Retrieve and deduplicate semantic candidates without final scoring."""

    if not query.strip():
        return []
    limit = SETTINGS.top_k_matches if top_k is None else top_k
    if limit <= 0:
        raise ValueError("top_k must be positive.")
    raw_candidates = list(retriever.search(query, limit))
    return _deduplicate_candidates(raw_candidates)[:limit]


def score_candidates(
    paragraph: str,
    candidates: Sequence[CandidateInput],
    *,
    min_score: float | None = None,
    ngram_size: int | None = None,
) -> list[ScoredMatch]:
    """Calculate lexical, phrase, and combined scores for retrieved candidates."""

    if not paragraph.strip():
        return []
    threshold = SETTINGS.min_match_score if min_score is None else min_score
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be in the range 0..1.")
    shingle_size = SETTINGS.ngram_size if ngram_size is None else ngram_size
    if shingle_size <= 0:
        raise ValueError("ngram_size must be positive.")

    matches: list[ScoredMatch] = []
    for candidate in _deduplicate_candidates(candidates):
        lexical = tfidf_similarity(paragraph, candidate.text)
        phrase_overlap = ngram_overlap(paragraph, candidate.text, size=shingle_size)
        score = combine_scores(
            lexical=lexical,
            semantic=candidate.semantic_score,
            ngram=phrase_overlap,
        )
        if score.final < threshold:
            continue
        matches.append(
            ScoredMatch(
                paragraph=paragraph,
                chunk_id=candidate.chunk_id,
                source_file=candidate.source_file,
                matched_text=candidate.text,
                lexical_similarity=score.lexical,
                semantic_similarity=score.semantic,
                ngram_overlap=score.ngram,
                final_score=score.final,
                risk=score.risk,
                reason=score.reason,
                source_path=candidate.source_path,
                word_count=candidate.word_count,
                page=candidate.page,
            )
        )

    return sorted(
        matches,
        key=lambda match: (
            -match.final_score,
            -match.semantic_similarity,
            match.source_file.casefold(),
            match.chunk_id,
        ),
    )


def analyze_paragraph(
    paragraph: str,
    retriever: CandidateRetriever,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    ngram_size: int | None = None,
) -> ParagraphAnalysis:
    """Run semantic candidate retrieval followed by final multi-signal scoring."""

    candidates = retrieve_candidates(paragraph, retriever, top_k=top_k)
    matches = score_candidates(
        paragraph,
        candidates,
        min_score=min_score,
        ngram_size=ngram_size,
    )
    return ParagraphAnalysis(
        paragraph=paragraph,
        candidates_retrieved=len(candidates),
        matches=tuple(matches),
    )
