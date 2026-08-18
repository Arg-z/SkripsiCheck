from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.similarity_engine import (
    SourceIndexerRetriever,
    analyze_paragraph,
    retrieve_candidates,
    score_candidates,
)


@dataclass(frozen=True)
class FakeCandidate:
    chunk_id: str
    source_file: str
    text: str
    semantic_score: float
    source_path: str | None = None
    word_count: int | None = None
    page: int | None = None


class FakeRetriever:
    def __init__(self, candidates: list[FakeCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[FakeCandidate]:
        self.calls.append((query, top_k))
        return self.candidates[:top_k]


def candidate(
    chunk_id: str,
    source_file: str,
    text: str,
    semantic_score: float,
) -> FakeCandidate:
    return FakeCandidate(
        chunk_id=chunk_id,
        source_file=source_file,
        source_path=f"references/{source_file}",
        text=text,
        word_count=len(text.split()),
        semantic_score=semantic_score,
    )


def test_retrieve_candidates_calls_retriever_and_deduplicates() -> None:
    retriever = FakeRetriever(
        [
            candidate("one", "source.pdf", "Versi relevan.", 0.70),
            candidate("one", "source.pdf", "Versi paling relevan.", 0.91),
            candidate("two", "other.txt", "Kandidat lain.", 0.60),
        ]
    )

    results = retrieve_candidates("paragraf uji", retriever, top_k=3)

    assert retriever.calls == [("paragraf uji", 3)]
    assert [result.chunk_id for result in results] == ["one", "two"]
    assert results[0].text == "Versi paling relevan."


def test_retrieve_candidates_does_not_search_for_empty_query() -> None:
    retriever = FakeRetriever([])

    assert retrieve_candidates("  ", retriever) == []
    assert retriever.calls == []


def test_score_candidates_combines_all_three_signals() -> None:
    paragraph = "Kadar kalsium memengaruhi kualitas cangkang telur puyuh."
    matches = score_candidates(
        paragraph,
        [candidate("same", "jurnal.pdf", paragraph, 0.90)],
        min_score=0.0,
    )

    assert len(matches) == 1
    match = matches[0]
    assert match.lexical_similarity == 1.0
    assert match.semantic_similarity == 0.90
    assert match.ngram_overlap == 1.0
    assert match.final_score == pytest.approx(0.35 + 0.45 * 0.90 + 0.20)
    assert match.risk == "VERY HIGH"


def test_score_candidates_filters_and_sorts_by_combined_score() -> None:
    paragraph = "Kalsium membantu pembentukan cangkang telur yang kuat."
    candidates = [
        candidate(
            "semantic-only",
            "parafrase.txt",
            "Mineral tertentu mendukung terbentuknya kulit telur yang kokoh.",
            0.90,
        ),
        candidate("exact", "kutipan.pdf", paragraph, 0.98),
        candidate(
            "irrelevant",
            "komputer.docx",
            "Enkripsi melindungi paket jaringan komputer.",
            0.05,
        ),
    ]

    matches = score_candidates(paragraph, candidates, min_score=0.40)

    assert [match.chunk_id for match in matches] == ["exact", "semantic-only"]
    assert all(match.final_score >= 0.40 for match in matches)


def test_score_candidates_deduplicates_direct_input() -> None:
    paragraph = "Metode penelitian menggunakan survei kuantitatif."
    duplicates = [
        candidate("chunk-1", "metode.pdf", paragraph, 0.75),
        candidate("chunk-1", "metode.pdf", paragraph, 0.95),
    ]

    matches = score_candidates(paragraph, duplicates, min_score=0.0)

    assert len(matches) == 1
    assert matches[0].semantic_similarity == 0.95


def test_analyze_paragraph_keeps_retrieval_separate_from_scoring() -> None:
    paragraph = "Analisis dilakukan terhadap data hasil wawancara."
    retriever = FakeRetriever(
        [candidate("chunk-7", "penelitian.pdf", paragraph, 0.93)]
    )

    result = analyze_paragraph(paragraph, retriever, top_k=7, min_score=0.0)

    assert retriever.calls == [(paragraph, 7)]
    assert result.paragraph == paragraph
    assert result.candidates_retrieved == 1
    assert len(result.matches) == 1
    assert result.matches[0].source_file == "penelitian.pdf"


def test_source_indexer_adapter_uses_search_similar_chunks() -> None:
    class FakeIndexer:
        def search_similar_chunks(
            self, query: str, top_k: int | None = None
        ) -> list[FakeCandidate]:
            assert query == "contoh query"
            assert top_k == 4
            return [candidate("adapted", "source.txt", query, 0.88)]

    adapter = SourceIndexerRetriever(FakeIndexer())

    results = adapter.search("contoh query", 4)

    assert results[0].chunk_id == "adapted"


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf")])
def test_non_finite_semantic_score_is_rejected(invalid_score: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        score_candidates(
            "query",
            [candidate("bad", "source.txt", "source", invalid_score)],
            min_score=0.0,
        )
