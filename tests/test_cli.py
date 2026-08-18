from __future__ import annotations

from pathlib import Path

import pytest

from app import cli
from app.core.indexer import IndexBuildResult, IndexBuildIssue, SemanticMatch


class FakeIndexer:
    instances: list["FakeIndexer"] = []

    def __init__(self, *, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir
        self.calls: list[tuple[str, object]] = []
        self.__class__.instances.append(self)

    def _result(self, sources: Path, progress_callback: object) -> IndexBuildResult:
        self.calls.append(("build", sources))
        if callable(progress_callback):
            progress_callback(1, 1, sources / "source.txt")
        output = self.index_dir or Path("data/index")
        return IndexBuildResult(
            source_count=1,
            chunks_indexed=2,
            embedding_dimension=384,
            index_path=output / "sources.faiss",
            metadata_path=output / "metadata.json",
            index_info_path=output / "index_info.json",
            skipped_files=(IndexBuildIssue("broken.pdf", "invalid PDF"),),
        )

    def build_index(self, sources: Path, *, progress_callback: object = None) -> IndexBuildResult:
        return self._result(sources, progress_callback)

    def rebuild_index(self, sources: Path, *, progress_callback: object = None) -> IndexBuildResult:
        self.calls.append(("rebuild", sources))
        return self._result(sources, progress_callback)

    def search_similar_chunks(
        self,
        query: str,
        top_k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[SemanticMatch]:
        self.calls.append(("search", (query, top_k, min_score)))
        return [
            SemanticMatch(
                chunk_id="chunk-1",
                source_file="jurnal.pdf",
                source_path="references/jurnal.pdf",
                text="Kalsium memperkuat cangkang telur.",
                word_count=5,
                semantic_score=0.91,
            )
        ]


@pytest.fixture(autouse=True)
def fake_indexer(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeIndexer.instances.clear()
    monkeypatch.setattr(cli, "SourceIndexer", FakeIndexer)


def test_index_command_prints_progress_and_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["index", "references", "--index-dir", "custom-index"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[1/1] source.txt" in output
    assert "Chunks indexed: 2" in output
    assert "Embedding dimension: 384" in output
    assert "broken.pdf" in output


def test_rebuild_command_uses_rebuild_method() -> None:
    assert cli.main(["rebuild-index", "references"]) == 0
    assert FakeIndexer.instances[0].calls[0][0] == "rebuild"


def test_search_command_prints_semantic_match(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["search", "kualitas cangkang", "--top-k", "3"]) == 0
    output = capsys.readouterr().out
    assert "jurnal.pdf" in output
    assert "91.0%" in output
    assert "chunk-1" in output
