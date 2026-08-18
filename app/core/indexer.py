"""Local source-document indexing and FAISS candidate retrieval.

The module deliberately keeps vector retrieval separate from final similarity
scoring.  FAISS returns likely source chunks; lexical, n-gram, and combined
scoring are performed by the analysis/service layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import faiss
import numpy as np

from app.config import SETTINGS
from app.core.chunker import chunk_document
from app.core.cleaner import clean_text
from app.core.extractor import DocumentExtractionError, SUPPORTED_EXTENSIONS, extract_text

INDEX_SCHEMA_VERSION = 2
INDEX_FILENAME = "sources.faiss"
METADATA_FILENAME = "metadata.json"
INDEX_INFO_FILENAME = "index_info.json"


class SourceIndexError(RuntimeError):
    """Base error for source indexing and retrieval failures."""


class SourceFolderError(SourceIndexError):
    """Raised when a source folder cannot be indexed."""


class NoSimilarityIndexError(SourceIndexError):
    """Raised when a persisted similarity index has not been created yet."""


class IndexIntegrityError(SourceIndexError):
    """Raised when persisted index files are invalid or out of sync."""


@runtime_checkable
class SemanticService(Protocol):
    """The small embedding-service contract required by the indexer."""

    @property
    def model_name(self) -> str:
        """Return the identifier used to generate embeddings."""

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode multiple texts as a two-dimensional float array."""

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one text as a one-dimensional float array."""


ProgressCallback = Callable[[int, int, Path], None]


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """Metadata whose list position corresponds to a FAISS vector position."""

    chunk_id: str
    source_file: str
    source_path: str
    text: str
    word_count: int
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> SourceChunk:
        """Validate and deserialize one persisted chunk."""

        if not isinstance(value, dict):
            raise IndexIntegrityError("Chunk metadata must be a JSON object.")
        required = ("chunk_id", "source_file", "source_path", "text", "word_count")
        missing = [field for field in required if field not in value]
        if missing:
            raise IndexIntegrityError(
                f"Chunk metadata is missing required fields: {', '.join(missing)}."
            )
        strings = {field: value[field] for field in required[:-1]}
        if any(not isinstance(item, str) or not item for item in strings.values()):
            raise IndexIntegrityError("Chunk text metadata fields must be non-empty strings.")
        word_count = value["word_count"]
        if isinstance(word_count, bool) or not isinstance(word_count, int) or word_count < 1:
            raise IndexIntegrityError("Chunk word_count must be a positive integer.")
        page = value.get("page")
        if page is not None and (
            isinstance(page, bool) or not isinstance(page, int) or page < 1
        ):
            raise IndexIntegrityError("Chunk page must be null or a positive integer.")
        return cls(
            chunk_id=strings["chunk_id"],
            source_file=strings["source_file"],
            source_path=strings["source_path"],
            text=strings["text"],
            word_count=word_count,
            page=page,
        )


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """A change-detection fingerprint retained for future incremental indexing."""

    source_path: str
    relative_path: str
    file_size: int
    modified_time_ns: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IndexBuildIssue:
    """A source that was skipped without aborting the remaining index build."""

    source_file: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Summary returned after a successful full index build."""

    source_count: int
    chunks_indexed: int
    embedding_dimension: int
    index_path: Path
    metadata_path: Path
    index_info_path: Path
    skipped_files: tuple[IndexBuildIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """One semantic candidate returned by FAISS."""

    chunk_id: str
    source_file: str
    source_path: str
    text: str
    word_count: int
    semantic_score: float
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_semantic_service() -> SemanticService:
    # Imported lazily so importing the index metadata types never loads a model.
    from app.core.semantic_similarity import get_semantic_service

    return get_semantic_service()


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """Return contiguous, normalized float32 rows suitable for IndexFlatIP."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise SourceIndexError("Embedding service must return a two-dimensional array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise SourceIndexError("Embedding service returned no usable vectors.")
    if not np.isfinite(array).all():
        raise SourceIndexError("Embedding service returned a non-finite value.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise SourceIndexError("Embedding service returned a zero-length source vector.")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _normalize_query(embedding: np.ndarray, expected_dimension: int) -> np.ndarray:
    query = np.asarray(embedding, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        raise SourceIndexError("A query embedding must contain exactly one vector.")
    if query.shape[1] != expected_dimension:
        raise IndexIntegrityError(
            "Query embedding dimension does not match the persisted FAISS index "
            f"({query.shape[1]} != {expected_dimension}). Rebuild the index with the active model."
        )
    if not np.isfinite(query).all():
        raise SourceIndexError("Query embedding contains a non-finite value.")
    norm = float(np.linalg.norm(query))
    if norm <= 0.0:
        raise SourceIndexError("Cannot search with an empty or zero-length query embedding.")
    return np.ascontiguousarray(query / norm, dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(path: Path, root: Path) -> SourceFingerprint:
    stat = path.stat()
    return SourceFingerprint(
        source_path=str(path),
        relative_path=path.relative_to(root).as_posix(),
        file_size=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
        sha256=_file_sha256(path),
    )


def _chunk_identifier(relative_path: str, paragraph_number: int, text: str) -> str:
    value = f"{relative_path}\0{paragraph_number}\0{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


class SourceIndexer:
    """Build, persist, load, and query a local FAISS source index."""

    def __init__(
        self,
        semantic_service: SemanticService | None = None,
        *,
        index_dir: str | Path | None = None,
    ) -> None:
        self.semantic_service = semantic_service or _default_semantic_service()
        self.index_dir = Path(index_dir if index_dir is not None else SETTINGS.index_dir)
        self._index: faiss.Index | None = None
        self._chunks: list[SourceChunk] = []
        self._index_info: dict[str, Any] = {}

    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / METADATA_FILENAME

    @property
    def index_info_path(self) -> Path:
        return self.index_dir / INDEX_INFO_FILENAME

    @property
    def chunks(self) -> tuple[SourceChunk, ...]:
        """Loaded chunk metadata, exposed as an immutable view."""

        return tuple(self._chunks)

    @property
    def index_info(self) -> dict[str, Any]:
        """Return a copy of persisted index information."""

        return dict(self._index_info)

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    def _discover_sources(self, source_folder: str | Path) -> tuple[Path, list[Path]]:
        raw_root = Path(source_folder).expanduser()
        try:
            root = raw_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SourceFolderError(f"Source folder not found: {raw_root}") from exc
        if not root.is_dir():
            raise SourceFolderError(f"Source path is not a folder: {root}")
        sources = sorted(
            (
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        if not sources:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise SourceFolderError(
                f"No supported source documents found in {root}. Expected: {allowed}."
            )
        return root, sources

    def build_index(
        self,
        source_folder: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexBuildResult:
        """Build a full index from all supported documents below a folder.

        Corrupt and text-empty documents are recorded as skipped files.  The
        build succeeds when at least one source produces one chunk.
        """

        root, sources = self._discover_sources(source_folder)
        chunks: list[SourceChunk] = []
        fingerprints: list[SourceFingerprint] = []
        issues: list[IndexBuildIssue] = []
        indexed_source_count = 0

        for current, source_path in enumerate(sources, start=1):
            if progress_callback is not None:
                progress_callback(current, len(sources), source_path)
            try:
                extracted = extract_text(source_path)
                paragraphs = [paragraph for paragraph, _ in chunk_document(clean_text(extracted))]
            except (DocumentExtractionError, OSError, RuntimeError, ValueError) as exc:
                issues.append(IndexBuildIssue(source_path.name, str(exc)))
                continue
            if not paragraphs:
                issues.append(IndexBuildIssue(source_path.name, "Document produced zero text chunks."))
                continue

            relative_path = source_path.relative_to(root).as_posix()
            for paragraph_number, paragraph in enumerate(paragraphs, start=1):
                chunks.append(
                    SourceChunk(
                        chunk_id=_chunk_identifier(relative_path, paragraph_number, paragraph),
                        source_file=source_path.name,
                        source_path=str(source_path),
                        text=paragraph,
                        word_count=len(paragraph.split()),
                    )
                )
            fingerprints.append(_source_fingerprint(source_path, root))
            indexed_source_count += 1

        if not chunks:
            details = "; ".join(
                f"{issue.source_file}: {issue.reason}" for issue in issues[:3]
            )
            suffix = f" Details: {details}" if details else ""
            raise SourceIndexError(
                "No text chunks could be indexed from the source folder." + suffix
            )

        embeddings = _normalize_rows(
            self.semantic_service.encode_texts([chunk.text for chunk in chunks])
        )
        if embeddings.shape[0] != len(chunks):
            raise SourceIndexError(
                "Embedding count does not match source chunk count "
                f"({embeddings.shape[0]} != {len(chunks)})."
            )

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        model_name = str(getattr(self.semantic_service, "model_name", "unknown"))
        created_at = datetime.now(timezone.utc).isoformat()
        build_id = uuid4().hex
        metadata_payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "build_id": build_id,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "source_fingerprints": [fingerprint.to_dict() for fingerprint in fingerprints],
        }
        info_payload: dict[str, Any] = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": created_at,
            "source_root": str(root),
            "model_name": model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "vector_count": len(chunks),
            "source_count": indexed_source_count,
            "skipped_source_count": len(issues),
            "skipped_sources": [issue.to_dict() for issue in issues],
            "normalized_embeddings": True,
            "distance_metric": "inner_product_cosine",
            "score_interpretation": "Cosine similarity is clipped to the range 0..1.",
        }
        self._persist(index, metadata_payload, info_payload)
        self._index = index
        self._chunks = chunks
        self._index_info = info_payload
        return IndexBuildResult(
            source_count=indexed_source_count,
            chunks_indexed=len(chunks),
            embedding_dimension=int(embeddings.shape[1]),
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            index_info_path=self.index_info_path,
            skipped_files=tuple(issues),
        )

    def rebuild_index(
        self,
        source_folder: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexBuildResult:
        """Recreate the full index; incremental updates can be added later."""

        return self.build_index(source_folder, progress_callback=progress_callback)

    def _persist(
        self,
        index: faiss.Index,
        metadata_payload: dict[str, Any],
        info_payload: dict[str, Any],
    ) -> None:
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SourceIndexError(f"Could not create index directory {self.index_dir}: {exc}") from exc

        temporary_paths = {
            self.index_path: self.index_path.with_suffix(self.index_path.suffix + ".tmp"),
            self.metadata_path: self.metadata_path.with_suffix(".json.tmp"),
            self.index_info_path: self.index_info_path.with_suffix(".json.tmp"),
        }
        try:
            faiss.write_index(index, str(temporary_paths[self.index_path]))
            temporary_paths[self.metadata_path].write_text(
                json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            info_payload["index_sha256"] = _file_sha256(temporary_paths[self.index_path])
            info_payload["metadata_sha256"] = _file_sha256(
                temporary_paths[self.metadata_path]
            )
            temporary_paths[self.index_info_path].write_text(
                json.dumps(info_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            # Metadata is replaced before the index, making the index file the
            # final commit marker. Any interruption is caught by load validation.
            os.replace(temporary_paths[self.metadata_path], self.metadata_path)
            os.replace(temporary_paths[self.index_info_path], self.index_info_path)
            os.replace(temporary_paths[self.index_path], self.index_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SourceIndexError(f"Could not save similarity index: {exc}") from exc
        finally:
            for temporary_path in temporary_paths.values():
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load_index(self) -> SourceIndexer:
        """Load and validate all three persisted index artifacts."""

        paths = (self.index_path, self.metadata_path, self.index_info_path)
        existing = [path.exists() for path in paths]
        if not any(existing):
            raise NoSimilarityIndexError(
                "No similarity index found.\n\nRun:\n\n"
                "skripsicheck index ./references"
            )
        if not all(existing):
            missing = ", ".join(path.name for path, present in zip(paths, existing) if not present)
            raise IndexIntegrityError(
                f"Similarity index is incomplete; missing: {missing}. Rebuild the index."
            )

        try:
            index = faiss.read_index(str(self.index_path))
            metadata_payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            info_payload = json.loads(self.index_info_path.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            raise IndexIntegrityError(
                f"Could not load similarity index files. Rebuild the index: {exc}"
            ) from exc

        if not isinstance(metadata_payload, dict) or not isinstance(info_payload, dict):
            raise IndexIntegrityError("Index metadata and index info must be JSON objects.")
        if metadata_payload.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise IndexIntegrityError("Unsupported source metadata schema. Rebuild the index.")
        if info_payload.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise IndexIntegrityError("Unsupported index-info schema. Rebuild the index.")
        build_id = metadata_payload.get("build_id")
        if (
            not isinstance(build_id, str)
            or not build_id
            or build_id != info_payload.get("build_id")
        ):
            raise IndexIntegrityError(
                "FAISS index metadata and index info are out of sync. Rebuild the index."
            )
        expected_index_hash = info_payload.get("index_sha256")
        expected_metadata_hash = info_payload.get("metadata_sha256")
        if (
            not isinstance(expected_index_hash, str)
            or len(expected_index_hash) != 64
            or not isinstance(expected_metadata_hash, str)
            or len(expected_metadata_hash) != 64
            or _file_sha256(self.index_path) != expected_index_hash
            or _file_sha256(self.metadata_path) != expected_metadata_hash
        ):
            raise IndexIntegrityError(
                "FAISS index artifacts are corrupt or out of sync. Rebuild the index."
            )
        raw_chunks = metadata_payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise IndexIntegrityError("Source metadata does not contain a valid chunks list.")
        chunks = [SourceChunk.from_dict(value) for value in raw_chunks]
        if index.ntotal != len(chunks):
            raise IndexIntegrityError(
                "FAISS index and source metadata are out of sync "
                f"({index.ntotal} vectors, {len(chunks)} chunks). Rebuild the index."
            )
        if not chunks:
            raise IndexIntegrityError("The persisted source index contains no chunks.")

        expected_dimension = info_payload.get("embedding_dimension")
        expected_count = info_payload.get("vector_count")
        if (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension != index.d
        ):
            raise IndexIntegrityError("FAISS index dimension and index info are out of sync.")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != index.ntotal
        ):
            raise IndexIntegrityError("FAISS vector count and index info are out of sync.")
        persisted_model = info_payload.get("model_name")
        active_model = str(getattr(self.semantic_service, "model_name", "unknown"))
        if not isinstance(persisted_model, str) or not persisted_model:
            raise IndexIntegrityError("Index info does not contain a valid model name.")
        if active_model != "unknown" and persisted_model != active_model:
            raise IndexIntegrityError(
                "The index was created with a different semantic model "
                f"({persisted_model!r} != {active_model!r}). Rebuild the index."
            )

        self._index = index
        self._chunks = chunks
        self._index_info = info_payload
        return self

    def delete_index(self) -> bool:
        """Delete only known index artifacts and clear in-memory state."""

        removed = False
        for path in (self.index_path, self.metadata_path, self.index_info_path):
            try:
                if path.exists():
                    path.unlink()
                    removed = True
            except OSError as exc:
                raise SourceIndexError(f"Could not delete index artifact {path}: {exc}") from exc
        self._index = None
        self._chunks = []
        self._index_info = {}
        return removed

    def retrieve_candidates(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[SemanticMatch]:
        """Retrieve semantic candidates without performing final scoring."""

        requested = SETTINGS.top_k_matches if top_k is None else top_k
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("top_k must be a positive integer.")
        threshold = SETTINGS.min_semantic_score if min_score is None else min_score
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("min_score must be in the range 0..1.")
        if self._index is None:
            self.load_index()
        assert self._index is not None
        query = _normalize_query(query_embedding, self._index.d)

        # A small over-fetch lets us suppress duplicated chunks without scanning
        # the complete corpus during normal candidate retrieval.
        candidate_count = min(self._index.ntotal, max(requested, requested * 4))
        similarities, positions = self._index.search(query, candidate_count)
        matches: list[SemanticMatch] = []
        seen_chunks: set[tuple[str, str]] = set()
        seen_text: set[tuple[str, str]] = set()
        for raw_score, position in zip(similarities[0], positions[0], strict=True):
            if position < 0:
                continue
            chunk = self._chunks[int(position)]
            chunk_key = (chunk.source_file, chunk.chunk_id)
            text_key = (
                chunk.source_path,
                hashlib.sha256(" ".join(chunk.text.split()).casefold().encode("utf-8")).hexdigest(),
            )
            if chunk_key in seen_chunks or text_key in seen_text:
                continue
            seen_chunks.add(chunk_key)
            seen_text.add(text_key)
            # Normalized inner product is cosine similarity in [-1, 1]. For
            # student-facing scoring we treat negative similarity as zero.
            score = min(1.0, max(0.0, float(raw_score)))
            if not math.isfinite(score):
                raise IndexIntegrityError("FAISS returned a non-finite similarity score.")
            if score < threshold:
                continue
            matches.append(
                SemanticMatch(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    source_path=chunk.source_path,
                    text=chunk.text,
                    word_count=chunk.word_count,
                    semantic_score=score,
                    page=chunk.page,
                )
            )
            if len(matches) >= requested:
                break
        return matches

    def search_similar_chunks(
        self,
        query: str,
        top_k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[SemanticMatch]:
        """Encode a paragraph and return its closest unique source chunks."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            return []
        embedding = self.semantic_service.encode_text(query)
        return self.retrieve_candidates(embedding, top_k=top_k, min_score=min_score)

    def search(self, query: str, top_k: int | None = None) -> list[SemanticMatch]:
        """Alias compatible with candidate-retriever service protocols."""

        return self.search_similar_chunks(query, top_k=top_k)


def build_source_index(
    source_folder: str | Path,
    *,
    semantic_service: SemanticService | None = None,
    index_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IndexBuildResult:
    """Convenience wrapper for a full source index build."""

    return SourceIndexer(semantic_service, index_dir=index_dir).build_index(
        source_folder, progress_callback=progress_callback
    )


def rebuild_source_index(
    source_folder: str | Path,
    *,
    semantic_service: SemanticService | None = None,
    index_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IndexBuildResult:
    """Convenience wrapper documenting that MVP indexing is a full rebuild."""

    return SourceIndexer(semantic_service, index_dir=index_dir).rebuild_index(
        source_folder, progress_callback=progress_callback
    )


def search_similar_chunks(
    query: str,
    top_k: int | None = None,
    *,
    min_score: float | None = None,
    semantic_service: SemanticService | None = None,
    index_dir: str | Path | None = None,
) -> list[SemanticMatch]:
    """Load the configured index and search it with a local embedding model."""

    return SourceIndexer(semantic_service, index_dir=index_dir).search_similar_chunks(
        query, top_k=top_k, min_score=min_score
    )
