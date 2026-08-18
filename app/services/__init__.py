"""Application services that orchestrate SkripsiCheck core components."""

from app.services.similarity_engine import (
    CandidateRetriever,
    ParagraphAnalysis,
    RetrievedCandidate,
    ScoredMatch,
    SourceIndexerRetriever,
    analyze_paragraph,
    retrieve_candidates,
    score_candidates,
)

__all__ = [
    "CandidateRetriever",
    "ParagraphAnalysis",
    "RetrievedCandidate",
    "ScoredMatch",
    "SourceIndexerRetriever",
    "analyze_paragraph",
    "retrieve_candidates",
    "score_candidates",
]
