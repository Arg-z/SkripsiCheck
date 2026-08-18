import pytest

from app.core.ngram_similarity import ngram_overlap, word_shingles


def test_ngram_overlap_detects_copied_phrase() -> None:
    query = "kadar kalsium sangat memengaruhi kualitas cangkang telur puyuh"
    source = "penelitian menunjukkan kadar kalsium sangat memengaruhi kualitas cangkang telur puyuh secara signifikan"
    assert ngram_overlap(query, source, size=4) == pytest.approx(1.0)


def test_ngram_overlap_is_zero_for_unrelated_text() -> None:
    assert ngram_overlap("satu dua tiga empat", "lima enam tujuh delapan", size=2) == 0.0


def test_short_query_can_match_inside_long_source() -> None:
    assert ngram_overlap("kualitas telur", "penelitian tentang kualitas telur puyuh", size=5) == 1.0


def test_invalid_shingle_size() -> None:
    with pytest.raises(ValueError):
        word_shingles("teks", size=0)
