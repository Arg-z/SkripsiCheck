import pytest

from app.core.lexical_similarity import is_exact_match, tfidf_against_sources, tfidf_similarity


def test_identical_text() -> None:
    text = "Analisis data dilakukan menggunakan metode kuantitatif."
    assert is_exact_match(text, text.upper())
    assert tfidf_similarity(text, text) == 1.0


def test_completely_different_text() -> None:
    score = tfidf_similarity(
        "Fotosintesis mengubah energi cahaya pada tumbuhan.",
        "Algoritma enkripsi menjaga kerahasiaan jaringan komputer.",
    )
    assert score == pytest.approx(0.0)


def test_tfidf_ranks_related_source_first() -> None:
    scores = tfidf_against_sources(
        "Kadar kalsium memengaruhi kualitas cangkang telur.",
        [
            "Kalsium berpengaruh terhadap kualitas cangkang telur puyuh.",
            "Keamanan jaringan menggunakan algoritma enkripsi.",
        ],
    )
    assert scores[0] > scores[1]

