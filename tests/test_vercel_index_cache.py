from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import pytest

from app.core.indexer import (
    INDEX_FILENAME,
    INDEX_INFO_FILENAME,
    METADATA_FILENAME,
    IndexIntegrityError,
    SourceIndexer,
)
from app.services.vercel_index_cache import (
    INDEX_ARTIFACT_FILENAMES,
    IndexArtifactDownloadError,
    IndexArtifactIntegrityError,
    VercelBlobIndexCache,
    VercelBlobIndexRetriever,
    VercelIndexCacheConfigurationError,
)


class FakeSemanticService:
    model_name = "vercel-index-cache-test-model"
    _terms = ("telur", "kalsium", "puyuh", "komputer")

    @classmethod
    def _encode(cls, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", text.casefold())
        values = [float(tokens.count(term)) for term in cls._terms]
        values.append(0.05)
        return np.asarray(values, dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._encode(text) for text in texts])

    def encode_text(self, text: str) -> np.ndarray:
        return self._encode(text)


class FakeBlobClient:
    def __init__(self, objects: dict[str, bytes], *, delay: float = 0.0) -> None:
        self.objects = objects
        self.delay = delay
        self.downloads: list[tuple[str, str]] = []
        self.closed = False
        self._lock = Lock()

    def download_file(
        self,
        url_or_path: str,
        local_path: str | os.PathLike[str],
        *,
        access: str = "public",
        timeout: float | None = None,
        overwrite: bool = True,
        create_parents: bool = True,
        token: str | None = None,
    ) -> str:
        del timeout, overwrite, create_parents, token
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.downloads.append((url_or_path, access))
        Path(local_path).write_bytes(self.objects[url_or_path])
        return str(local_path)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def semantic_service() -> FakeSemanticService:
    return FakeSemanticService()


@pytest.fixture
def remote_artifacts(
    tmp_path: Path, semantic_service: FakeSemanticService
) -> dict[str, bytes]:
    sources = tmp_path / "references"
    sources.mkdir()
    (sources / "jurnal.txt").write_text(
        "Kalsium membantu memperkuat cangkang telur puyuh.\n\n"
        "Komputer memproses perangkat lunak.",
        encoding="utf-8",
    )
    origin = tmp_path / "built-index"
    SourceIndexer(semantic_service, index_dir=origin).build_index(sources)
    prefix = "indexes/test-build"
    return {
        f"{prefix}/{filename}": (origin / filename).read_bytes()
        for filename in INDEX_ARTIFACT_FILENAMES
    }


def make_cache(
    tmp_path: Path,
    client: FakeBlobClient,
    semantic_service: FakeSemanticService,
) -> VercelBlobIndexCache:
    return VercelBlobIndexCache(
        prefix="indexes/test-build",
        cache_root=tmp_path / "cache",
        semantic_service=semantic_service,
        client=client,
    )


def test_downloads_private_bundle_once_and_returns_valid_index(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
) -> None:
    client = FakeBlobClient(remote_artifacts)
    cache = make_cache(tmp_path, client, semantic_service)

    first = cache.ensure_index_dir()
    second = cache.ensure_index_dir()

    assert first == second
    assert {path.name for path in first.iterdir()} == set(INDEX_ARTIFACT_FILENAMES)
    assert len(client.downloads) == 3
    assert all(access == "private" for _, access in client.downloads)
    loaded = SourceIndexer(semantic_service, index_dir=first).load_index()
    assert loaded.index_info["vector_count"] == 2
    assert not list(cache.namespace_dir.glob(".download-*"))


def test_threaded_ensure_downloads_each_artifact_only_once(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
) -> None:
    client = FakeBlobClient(remote_artifacts, delay=0.01)
    caches = [make_cache(tmp_path, client, semantic_service) for _ in range(6)]

    with ThreadPoolExecutor(max_workers=6) as executor:
        paths = list(executor.map(lambda cache: cache.ensure_index_dir(), caches))

    assert len(set(paths)) == 1
    assert len(client.downloads) == 3


def test_incomplete_warm_cache_is_replaced_from_remote(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
) -> None:
    client = FakeBlobClient(remote_artifacts)
    cache = make_cache(tmp_path, client, semantic_service)
    cache.index_dir.mkdir(parents=True)
    (cache.index_dir / METADATA_FILENAME).write_text("{}", encoding="utf-8")

    index_dir = cache.ensure_index_dir()

    assert len(client.downloads) == 3
    assert {path.name for path in index_dir.iterdir()} == set(INDEX_ARTIFACT_FILENAMES)


def test_source_indexer_integrity_failure_is_wrapped_and_not_published(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
) -> None:
    corrupt = dict(remote_artifacts)
    metadata_path = "indexes/test-build/metadata.json"
    corrupt[metadata_path] = corrupt[metadata_path] + b"corrupt"
    cache = make_cache(tmp_path, FakeBlobClient(corrupt), semantic_service)

    with pytest.raises(IndexArtifactIntegrityError) as captured:
        cache.ensure_index_dir()

    assert isinstance(captured.value.__cause__, IndexIntegrityError)
    assert not cache.index_dir.exists()
    assert not list(cache.namespace_dir.glob(".download-*"))


def test_missing_remote_artifact_has_informative_secret_free_error(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
) -> None:
    incomplete = dict(remote_artifacts)
    incomplete.pop("indexes/test-build/index_info.json")
    cache = make_cache(tmp_path, FakeBlobClient(incomplete), semantic_service)

    with pytest.raises(IndexArtifactDownloadError, match="index_info.json") as captured:
        cache.ensure_index_dir()

    assert "token" not in str(captured.value).casefold()
    assert not cache.index_dir.exists()


def test_client_and_credential_are_lazy(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_BLOB_TOKEN", raising=False)
    created_with: list[str] = []

    def factory(token: str) -> FakeBlobClient:
        created_with.append(token)
        return FakeBlobClient(remote_artifacts)

    cache = VercelBlobIndexCache(
        prefix="indexes/test-build",
        cache_root=tmp_path / "cache",
        token_env="TEST_BLOB_TOKEN",
        semantic_service=semantic_service,
        client_factory=factory,
    )
    assert created_with == []
    with pytest.raises(VercelIndexCacheConfigurationError, match="TEST_BLOB_TOKEN"):
        cache.ensure_index_dir()
    assert created_with == []

    monkeypatch.setenv("TEST_BLOB_TOKEN", "unit-test-secret")
    cache.ensure_index_dir()
    assert created_with == ["unit-test-secret"]


@pytest.mark.parametrize("prefix", ["", "/", "../outside", "index/../../outside"])
def test_rejects_unsafe_blob_prefix(prefix: str, tmp_path: Path) -> None:
    with pytest.raises(VercelIndexCacheConfigurationError, match="safe relative"):
        VercelBlobIndexCache(prefix=prefix, cache_root=tmp_path)


def test_candidate_retriever_is_lazy_and_searches_cached_index(
    tmp_path: Path,
    remote_artifacts: dict[str, bytes],
    semantic_service: FakeSemanticService,
) -> None:
    client = FakeBlobClient(remote_artifacts)
    retriever = VercelBlobIndexRetriever(
        make_cache(tmp_path, client, semantic_service)
    )

    assert not retriever.is_loaded
    assert client.downloads == []
    matches = retriever.search("kalsium pada telur puyuh", top_k=1)

    assert retriever.is_loaded
    assert matches[0].source_file == "jurnal.txt"
    assert "Kalsium" in matches[0].text
    assert len(client.downloads) == 3


def test_expected_artifact_names_stay_aligned_with_source_indexer() -> None:
    assert INDEX_ARTIFACT_FILENAMES == (
        INDEX_FILENAME,
        METADATA_FILENAME,
        INDEX_INFO_FILENAME,
    )
