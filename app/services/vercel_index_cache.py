"""Warm-instance cache for FAISS artifacts stored in Private Vercel Blob.

Only the three persisted artifacts produced by :class:`SourceIndexer` are
downloaded.  They are staged in the platform temporary directory, validated by
``SourceIndexer.load_index()``, and then published with an atomic directory
rename.  Consequently, readers never observe a partially downloaded bundle.

The Blob credential is read lazily and is never included in an exception or
log message.  Local development remains unchanged unless this adapter is
explicitly selected by the application container.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Protocol
from uuid import uuid4

from app.core.indexer import (
    INDEX_FILENAME,
    INDEX_INFO_FILENAME,
    METADATA_FILENAME,
    IndexIntegrityError,
    SemanticService,
    SourceIndexError,
    SourceIndexer,
)
from app.services.similarity_engine import SemanticCandidate

DEFAULT_INDEX_BLOB_PREFIX = "indexes/current"
DEFAULT_INDEX_BLOB_PREFIX_ENV = "SKRIPSICHECK_BLOB_INDEX_PREFIX"
DEFAULT_INDEX_CACHE_DIR_ENV = "SKRIPSICHECK_INDEX_CACHE_DIR"
DEFAULT_BLOB_TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"
INDEX_ARTIFACT_FILENAMES = (
    INDEX_FILENAME,
    METADATA_FILENAME,
    INDEX_INFO_FILENAME,
)


class VercelIndexCacheError(SourceIndexError):
    """Base error for remote index configuration, download, or caching."""


class VercelIndexCacheConfigurationError(VercelIndexCacheError):
    """Raised when Private Vercel Blob is not configured correctly."""


class IndexArtifactDownloadError(VercelIndexCacheError):
    """Raised when one of the required remote artifacts cannot be downloaded."""


class IndexArtifactIntegrityError(VercelIndexCacheError):
    """Raised when SourceIndexer rejects a downloaded artifact bundle."""


class BlobIndexClientProtocol(Protocol):
    """Small subset of the Vercel Blob client required by the cache."""

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
    ) -> str: ...

    def close(self) -> None: ...


BlobClientFactory = Callable[[str], BlobIndexClientProtocol]


def _default_client_factory(token: str) -> BlobIndexClientProtocol:
    try:
        from vercel.blob import BlobClient
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise VercelIndexCacheConfigurationError(
            "Private index caching requires the 'vercel' Python package."
        ) from exc
    return BlobClient(token=token)


def _normalized_prefix(value: str) -> str:
    prefix = value.strip().strip("/")
    candidate = PurePosixPath(prefix)
    if (
        not prefix
        or candidate.is_absolute()
        or candidate in {PurePosixPath("."), PurePosixPath("..")}
        or ".." in candidate.parts
    ):
        raise VercelIndexCacheConfigurationError(
            "The index Blob prefix must be a safe relative path."
        )
    return candidate.as_posix()


_PATH_LOCKS: dict[str, RLock] = {}
_PATH_LOCKS_GUARD = Lock()


def _lock_for(path: Path) -> RLock:
    """Return a process-wide lock shared by adapters targeting one cache path."""

    key = os.path.normcase(str(path.absolute()))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, RLock())


class VercelBlobIndexCache:
    """Lazily materialize one coherent FAISS index bundle under temporary storage.

    A prefix identifies a remotely published generation. To publish a new
    generation safely, upload all three files below another prefix and update
    ``SKRIPSICHECK_INDEX_BLOB_PREFIX`` in the deployment environment.
    """

    def __init__(
        self,
        *,
        prefix: str | None = None,
        cache_root: str | Path | None = None,
        token_env: str = DEFAULT_BLOB_TOKEN_ENV,
        semantic_service: SemanticService | None = None,
        client: BlobIndexClientProtocol | None = None,
        client_factory: BlobClientFactory = _default_client_factory,
    ) -> None:
        configured_prefix = prefix
        if configured_prefix is None:
            configured_prefix = os.getenv(
                DEFAULT_INDEX_BLOB_PREFIX_ENV, DEFAULT_INDEX_BLOB_PREFIX
            )
        self.prefix = _normalized_prefix(configured_prefix)
        if not token_env.strip():
            raise VercelIndexCacheConfigurationError("token_env cannot be empty.")

        configured_cache_root = cache_root
        if configured_cache_root is None:
            configured_cache_root = os.getenv(DEFAULT_INDEX_CACHE_DIR_ENV)
        self.cache_root = Path(
            configured_cache_root
            or (Path(tempfile.gettempdir()) / "skripsicheck-index-cache")
        )
        namespace = hashlib.sha256(self.prefix.encode("utf-8")).hexdigest()[:16]
        self.namespace_dir = self.cache_root / namespace
        self.index_dir = self.namespace_dir / "current"
        self.token_env = token_env
        self.semantic_service = semantic_service
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None
        self._ready_indexer: SourceIndexer | None = None
        self._lock = _lock_for(self.index_dir)

    def ensure_index_dir(self) -> Path:
        """Return a complete local index directory, downloading it once if needed."""

        self.load_indexer()
        return self.index_dir

    def load_indexer(self) -> SourceIndexer:
        """Return a loaded, integrity-checked SourceIndexer for this warm instance."""

        with self._lock:
            if self._ready_indexer is not None and self.index_dir.is_dir():
                return self._ready_indexer

            if self.index_dir.is_dir():
                try:
                    self._ready_indexer = self._validated_indexer(self.index_dir)
                    return self._ready_indexer
                except SourceIndexError:
                    self._remove_known_cache_directory(self.index_dir)
            elif self.index_dir.exists():
                raise VercelIndexCacheError(
                    "The temporary similarity-index cache path is not a directory."
                )

            self._ready_indexer = self._download_and_publish()
            return self._ready_indexer

    def _download_and_publish(self) -> SourceIndexer:
        try:
            self.namespace_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VercelIndexCacheError(
                "Could not create temporary storage for the similarity index."
            ) from exc

        staging_dir = self.namespace_dir / f".download-{uuid4().hex}"
        try:
            staging_dir.mkdir()
            for filename in INDEX_ARTIFACT_FILENAMES:
                self._download_artifact(filename, staging_dir / filename)

            # SourceIndexer owns all cross-file checks: hashes, build id, schema,
            # model, vector dimensions, and metadata/vector counts.
            self._validated_downloaded_indexer(staging_dir)
            try:
                os.replace(staging_dir, self.index_dir)
            except OSError as exc:
                # Another worker process may have atomically published the same
                # prefix while this process downloaded it. Prefer that valid copy.
                if self.index_dir.is_dir():
                    try:
                        return self._validated_indexer(self.index_dir)
                    except IndexIntegrityError:
                        pass
                raise VercelIndexCacheError(
                    "Could not publish the downloaded similarity index atomically."
                ) from exc
            return self._validated_downloaded_indexer(self.index_dir)
        finally:
            if staging_dir.exists():
                self._remove_known_cache_directory(staging_dir)

    def _download_artifact(self, filename: str, target: Path) -> None:
        remote_path = f"{self.prefix}/{filename}"
        try:
            self._get_client().download_file(
                remote_path,
                target,
                access="private",
                overwrite=False,
                create_parents=False,
            )
        except VercelIndexCacheError:
            raise
        except Exception as exc:
            raise IndexArtifactDownloadError(
                f"Could not download required similarity-index artifact {filename!r} "
                "from Private Vercel Blob. Check that all three artifacts exist "
                "under the configured prefix."
            ) from exc
        if not target.is_file() or target.stat().st_size <= 0:
            raise IndexArtifactDownloadError(
                f"Private Vercel Blob returned an empty similarity-index artifact: "
                f"{filename}."
            )

    def _validated_indexer(self, index_dir: Path) -> SourceIndexer:
        return SourceIndexer(
            self.semantic_service, index_dir=index_dir
        ).load_index()

    def _validated_downloaded_indexer(self, index_dir: Path) -> SourceIndexer:
        """Translate only SourceIndexer validation failures into artifact errors."""

        try:
            return self._validated_indexer(index_dir)
        except VercelIndexCacheError:
            raise
        except SourceIndexError as exc:
            raise IndexArtifactIntegrityError(
                "The Private Vercel Blob similarity-index artifacts failed integrity "
                "validation. Upload sources.faiss, metadata.json, and index_info.json "
                "from the same index build."
            ) from exc

    def _get_client(self) -> BlobIndexClientProtocol:
        if self._client is not None:
            return self._client
        token = os.getenv(self.token_env)
        if not token:
            raise VercelIndexCacheConfigurationError(
                f"Private Vercel Blob requires the {self.token_env} environment variable."
            )
        try:
            self._client = self._client_factory(token)
        except VercelIndexCacheError:
            raise
        except Exception as exc:
            raise VercelIndexCacheConfigurationError(
                "Could not initialize the Private Vercel Blob client."
            ) from exc
        return self._client

    def _remove_known_cache_directory(self, path: Path) -> None:
        """Remove only a direct child managed inside this prefix's namespace."""

        try:
            namespace = self.namespace_dir.resolve()
            candidate = path.resolve()
            if candidate.parent != namespace or candidate == namespace:
                raise VercelIndexCacheError(
                    "Refusing to remove a path outside the index cache namespace."
                )
            shutil.rmtree(candidate)
        except VercelIndexCacheError:
            raise
        except OSError as exc:
            raise VercelIndexCacheError(
                "Could not clean an incomplete temporary similarity index."
            ) from exc

    def close(self) -> None:
        """Close a lazily-created SDK transport owned by this adapter."""

        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


class VercelBlobIndexRetriever:
    """CandidateRetriever that loads the remote index on its first search only."""

    def __init__(self, cache: VercelBlobIndexCache | None = None) -> None:
        self.cache = cache or get_vercel_index_cache()
        self._indexer: SourceIndexer | None = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._indexer is not None

    def search(self, query: str, top_k: int) -> Sequence[SemanticCandidate]:
        if self._indexer is None:
            with self._lock:
                if self._indexer is None:
                    self._indexer = self.cache.load_indexer()
        return self._indexer.search(query, top_k=top_k)


_DEFAULT_CACHE: VercelBlobIndexCache | None = None
_DEFAULT_CACHE_LOCK = Lock()


def get_vercel_index_cache() -> VercelBlobIndexCache:
    """Return the process-wide cache without contacting Blob or loading FAISS."""

    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        with _DEFAULT_CACHE_LOCK:
            if _DEFAULT_CACHE is None:
                _DEFAULT_CACHE = VercelBlobIndexCache()
    return _DEFAULT_CACHE


def build_vercel_index_retriever() -> VercelBlobIndexRetriever:
    """Factory suitable for ``build_container(retriever_factory=...)``."""

    return VercelBlobIndexRetriever(get_vercel_index_cache())
