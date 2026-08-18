import pytest

from app.core.scorer import combine_scores, risk_category


def test_combined_score_uses_configured_formula() -> None:
    result = combine_scores(lexical=0.82, semantic=0.91, ngram=0.76)
    expected = 0.35 * 0.82 + 0.45 * 0.91 + 0.20 * 0.76
    assert result.final == pytest.approx(expected)
    assert result.risk == "VERY HIGH"
    assert "semantic" in result.reason
    assert "phrase overlap" in result.reason


@pytest.mark.parametrize(
    ("score", "category"),
    [(0.0, "LOW"), (0.40, "MODERATE"), (0.60, "HIGH"), (0.80, "VERY HIGH")],
)
def test_risk_boundaries(score: float, category: str) -> None:
    assert risk_category(score) == category


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        combine_scores(0.5, 0.5, 0.5, lexical_weight=1, semantic_weight=1, ngram_weight=1)


def test_non_finite_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="range 0..1"):
        combine_scores(float("nan"), 0.5, 0.5)
