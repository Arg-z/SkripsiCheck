"""Transparent combination of independent similarity signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import SETTINGS


@dataclass(frozen=True, slots=True)
class SimilarityScore:
    lexical: float
    semantic: float
    ngram: float
    final: float
    risk: str
    reason: str


def risk_category(score: float) -> str:
    """Map a normalized combined score to the documented review category."""

    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("Score must be in the range 0..1.")
    if score < 0.40:
        return "LOW"
    if score < 0.60:
        return "MODERATE"
    if score < 0.80:
        return "HIGH"
    return "VERY HIGH"


def _reason(lexical: float, semantic: float, ngram: float) -> str:
    strong: list[str] = []
    if semantic >= 0.70:
        strong.append("high semantic similarity")
    if lexical >= 0.70:
        strong.append("high lexical similarity")
    if ngram >= 0.60:
        strong.append("high phrase overlap")
    if strong:
        return " and ".join(strong).capitalize() + "."
    if semantic >= 0.40 or lexical >= 0.40 or ngram >= 0.40:
        return "One or more similarity signals need manual review."
    return "No strong similarity signal was found."


def combine_scores(
    lexical: float,
    semantic: float,
    ngram: float,
    *,
    lexical_weight: float | None = None,
    semantic_weight: float | None = None,
    ngram_weight: float | None = None,
) -> SimilarityScore:
    """Combine three scores; weights are configurable and must sum to one."""

    signals = (lexical, semantic, ngram)
    if any(not math.isfinite(score) or score < 0.0 or score > 1.0 for score in signals):
        raise ValueError("All similarity signals must be in the range 0..1.")
    weights = (
        SETTINGS.lexical_weight if lexical_weight is None else lexical_weight,
        SETTINGS.semantic_weight if semantic_weight is None else semantic_weight,
        SETTINGS.ngram_weight if ngram_weight is None else ngram_weight,
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights) or not 0.999 <= sum(weights) <= 1.001:
        raise ValueError("Weights must be non-negative and sum to 1.0.")
    final = sum(score * weight for score, weight in zip(signals, weights, strict=True))
    return SimilarityScore(
        lexical=lexical,
        semantic=semantic,
        ngram=ngram,
        final=final,
        risk=risk_category(final),
        reason=_reason(lexical, semantic, ngram),
    )
