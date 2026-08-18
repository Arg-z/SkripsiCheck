import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.core.indexer import (
    INDEX_FILENAME,
    INDEX_INFO_FILENAME,
    METADATA_FILENAME,
    IndexIntegrityError,
    NoSimilarityIndexError,
    SourceFolderError,
    SourceIndexer,
)


class FakeSemanticService:
    """Small deterministic embedder; tests never download a real model."""

    model_name = "deterministic-test-model"
    _terms = ("telur", "kalsium", "puyuh", "komputer", "hujan", "penelitian")

    @classmethod
    def _encode(cls, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", text.casefold())
        features = [float(tokens.count(term)) for term in cls._terms]
        # A small non-zero fallback dimension also exercises normalization.
        features.append(0.05 + (sum(map(ord, text)) % 7) / 100.0)
        return np.asarray(features, dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._encode(text) for text in texts])

    def encode_text(self, text: str) -> np.ndarray:
        return self._encode(text)


@pytest.fixture
def source_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "references"
    folder.mkdir()
    (folder / "telur.txt").write_text(
        "Kalsium membantu memperkuat cangkang telur puyuh.\n\n"
        "Penelitian mengukur kualitas telur setiap minggu.",
        encoding="utf-8",
    )
    (folder / "teknologi.txt").write_text(
        "Komputer menjalankan perangkat lunak untuk mengolah data.",
        encoding="utf-8",
    )
    return folder


def make_indexer(tmp_path: Path) -> SourceIndexer:
    return SourceIndexer(FakeSemanticService(), index_dir=tmp_path / "index")


def test_source_indexing_creates_faiss_and_metadata(
    tmp_path: Path, source_folder: Path
) -> None:
    indexer = make_indexer(tmp_path)

    result = indexer.build_index(source_folder)

    assert result.source_count == 2
    assert result.chunks_indexed == 3
    assert result.embedding_dimension == 7
    assert (tmp_path / "index" / INDEX_FILENAME).is_file()
    assert (tmp_path / "index" / METADATA_FILENAME).is_file()
    assert (tmp_path / "index" / INDEX_INFO_FILENAME).is_file()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["chunks"]) == 3
    assert len(metadata["source_fingerprints"]) == 2
    first = metadata["chunks"][0]
    assert {
        "chunk_id",
        "source_file",
        "source_path",
        "text",
        "word_count",
        "page",
    } <= first.keys()
    assert len(metadata["source_fingerprints"][0]["sha256"]) == 64
    info = json.loads(result.index_info_path.read_text(encoding="utf-8"))
    assert metadata["build_id"] == info["build_id"]
    assert len(info["index_sha256"]) == 64
    assert len(info["metadata_sha256"]) == 64


def test_saved_index_can_be_loaded_and_searched(
    tmp_path: Path, source_folder: Path
) -> None:
    make_indexer(tmp_path).build_index(source_folder)

    reopened = make_indexer(tmp_path).load_index()
    matches = reopened.search_similar_chunks("Manfaat kalsium bagi telur puyuh", top_k=2)

    assert reopened.is_loaded
    assert len(reopened.chunks) == 3
    assert len(matches) == 2
    assert matches[0].source_file == "telur.txt"
    assert "Kalsium" in matches[0].text
    assert 0.0 <= matches[0].semantic_score <= 1.0
    assert matches[0].semantic_score >= matches[1].semantic_score


def test_search_alias_uses_candidate_retrieval(
    tmp_path: Path, source_folder: Path
) -> None:
    indexer = make_indexer(tmp_path)
    indexer.build_index(source_folder)

    match = indexer.search("perangkat komputer", top_k=1)[0]

    assert match.source_file == "teknologi.txt"
    assert match.to_dict()["semantic_score"] == pytest.approx(match.semantic_score)


def test_build_reports_progress_and_skips_a_corrupt_document(
    tmp_path: Path, source_folder: Path
) -> None:
    (source_folder / "broken.pdf").write_text("not a PDF", encoding="utf-8")
    progress: list[tuple[int, int, str]] = []
    indexer = make_indexer(tmp_path)

    result = indexer.build_index(
        source_folder,
        progress_callback=lambda current, total, path: progress.append(
            (current, total, path.name)
        ),
    )

    assert [item[0] for item in progress] == [1, 2, 3]
    assert all(item[1] == 3 for item in progress)
    assert len(result.skipped_files) == 1
    assert result.skipped_files[0].source_file == "broken.pdf"
    assert "not PDF" in result.skipped_files[0].reason


def test_duplicate_chunks_from_one_source_are_not_returned_twice(tmp_path: Path) -> None:
    sources = tmp_path / "references"
    sources.mkdir()
    repeated = "Kalsium membantu kualitas telur puyuh."
    (sources / "duplicate.txt").write_text(
        f"{repeated}\n\n{repeated}\n\nKomputer mengolah data.", encoding="utf-8"
    )
    indexer = make_indexer(tmp_path)
    indexer.build_index(sources)

    matches = indexer.search_similar_chunks("kalsium telur puyuh", top_k=3)

    repeated_matches = [match for match in matches if match.text == repeated]
    assert len(repeated_matches) == 1


def test_missing_or_empty_index_has_informative_errors(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path)

    with pytest.raises(NoSimilarityIndexError, match="skripsicheck index"):
        indexer.load_index()
    with pytest.raises(SourceFolderError, match="not found"):
        indexer.build_index(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SourceFolderError, match="No supported source documents"):
        indexer.build_index(empty)


def test_load_detects_index_and_metadata_out_of_sync(
    tmp_path: Path, source_folder: Path
) -> None:
    indexer = make_indexer(tmp_path)
    result = indexer.build_index(source_folder)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    metadata["chunks"].pop()
    result.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexIntegrityError, match="out of sync"):
        make_indexer(tmp_path).load_index()


def test_blank_query_is_safe_and_top_k_is_validated(
    tmp_path: Path, source_folder: Path
) -> None:
    indexer = make_indexer(tmp_path)
    indexer.build_index(source_folder)

    assert indexer.search_similar_chunks("   ") == []
    with pytest.raises(ValueError, match="positive integer"):
        indexer.search_similar_chunks("telur", top_k=0)


def test_semantic_threshold_filters_weak_candidates(
    tmp_path: Path, source_folder: Path
) -> None:
    indexer = make_indexer(tmp_path)
    indexer.build_index(source_folder)

    matches = indexer.search_similar_chunks("kalsium telur puyuh", min_score=0.90)

    assert matches
    assert all(match.semantic_score >= 0.90 for match in matches)


def test_rebuild_and_delete_index(tmp_path: Path, source_folder: Path) -> None:
    indexer = make_indexer(tmp_path)
    first = indexer.build_index(source_folder)
    (source_folder / "tambahan.txt").write_text(
        "Hujan memengaruhi kelembapan tanah pertanian.", encoding="utf-8"
    )

    rebuilt = indexer.rebuild_index(source_folder)

    assert rebuilt.source_count == first.source_count + 1
    assert rebuilt.chunks_indexed == first.chunks_indexed + 1
    assert indexer.delete_index()
    assert not rebuilt.index_path.exists()
    assert not rebuilt.metadata_path.exists()
    assert not rebuilt.index_info_path.exists()
