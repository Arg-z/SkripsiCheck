"""Local sentence-embedding service and semantic cosine similarity.

The Sentence Transformer dependency and model are loaded only when embeddings
are first requested.  This keeps lightweight commands fast and also makes the
service straightforward to test without downloading a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.config import SETTINGS

FloatArray = NDArray[np.float32]


class SentenceEncoder(Protocol):
    """Small portion of ``SentenceTransformer`` used by this module."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> ArrayLike: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


ModelFactory = Callable[[str, str], SentenceEncoder]


def _default_model_factory(model_name: str, device: str) -> SentenceEncoder:
    """Import and construct SentenceTransformer only when it is first needed."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError(
            "Semantic similarity requires sentence-transformers. "
            "Install the project dependencies before building or searching an index."
        ) from exc

    return cast(SentenceEncoder, SentenceTransformer(model_name, device=device))


class SemanticSimilarityService:
    """Lazily load one multilingual embedding model and reuse it for encoding."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.model_name = SETTINGS.semantic_model if model_name is None else model_name
        self.batch_size = (
            getattr(SETTINGS, "embedding_batch_size", 32)
            if batch_size is None
            else batch_size
        )
        self.device = getattr(SETTINGS, "device", "cpu") if device is None else device
        if not self.model_name.strip():
            raise ValueError("Semantic model name cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError("Embedding batch size must be positive.")
        if not self.device.strip():
            raise ValueError("Embedding device cannot be empty.")
        self._model_factory = model_factory or _default_model_factory
        self._model: SentenceEncoder | None = None
        self._model_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether this service instance has already loaded its model."""

        return self._model is not None

    def _get_model(self) -> SentenceEncoder:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(self.model_name, self.device)
        return self._model

    def _embedding_dimension(self) -> int:
        dimension = self._get_model().get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise RuntimeError("Sentence Transformer reported an invalid embedding dimension.")
        return int(dimension)

    def encode_texts(self, texts: Sequence[str]) -> FloatArray:
        """Encode texts as normalized float32 rows; blank texts become zero rows.

        An empty collection does not trigger model loading and returns an empty
        ``(0, 0)`` array.  Blank items inside a non-empty collection are retained
        at their original positions so metadata and embeddings cannot drift.
        """

        text_list = list(texts)
        if any(not isinstance(text, str) for text in text_list):
            raise TypeError("All texts must be strings.")
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)

        nonempty_positions = [index for index, text in enumerate(text_list) if text.strip()]
        if not nonempty_positions:
            return np.zeros((len(text_list), self._embedding_dimension()), dtype=np.float32)

        model = self._get_model()
        nonempty_texts = [text_list[index] for index in nonempty_positions]
        raw_embeddings = model.encode(
            nonempty_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim == 1 and len(nonempty_texts) == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(nonempty_texts):
            raise RuntimeError("Sentence Transformer returned an unexpected embedding shape.")
        if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
            raise RuntimeError("Sentence Transformer returned invalid embeddings.")

        # Normalize defensively: injected/custom encoders may ignore the flag.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings),
            where=norms > 0.0,
        )
        result = np.zeros((len(text_list), embeddings.shape[1]), dtype=np.float32)
        result[nonempty_positions] = embeddings
        return np.ascontiguousarray(result, dtype=np.float32)

    def encode_text(self, text: str) -> FloatArray:
        """Encode one text and return a one-dimensional normalized embedding."""

        return self.encode_texts([text])[0]


# Concise compatibility name used by the indexing layer.
SemanticModelService = SemanticSimilarityService


_DEFAULT_SERVICE: SemanticSimilarityService | None = None
_DEFAULT_SERVICE_LOCK = Lock()


def get_semantic_service() -> SemanticSimilarityService:
    """Return the process-wide service without loading its model eagerly."""

    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = SemanticSimilarityService()
    return _DEFAULT_SERVICE


def encode_texts(texts: Sequence[str]) -> FloatArray:
    """Encode a collection with the reusable process-wide model service."""

    return get_semantic_service().encode_texts(texts)


def encode_text(text: str) -> FloatArray:
    """Encode one text with the reusable process-wide model service."""

    return get_semantic_service().encode_text(text)


def semantic_similarity(query_embedding: ArrayLike, source_embedding: ArrayLike) -> float:
    """Return cosine similarity as a user-facing score in the range 0..1.

    Native cosine values can range from -1 to 1.  Negative values indicate no
    useful semantic match for retrieval, so SkripsiCheck clips them to zero.
    Empty and zero-vector inputs also return zero.
    """

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    source = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
    if query.size == 0 or source.size == 0:
        return 0.0
    if query.shape != source.shape:
        raise ValueError("Semantic embeddings must have the same dimension.")
    if not np.isfinite(query).all() or not np.isfinite(source).all():
        raise ValueError("Semantic embeddings must contain only finite values.")

    denominator = float(np.linalg.norm(query) * np.linalg.norm(source))
    if denominator == 0.0:
        return 0.0
    cosine = float(np.dot(query, source) / denominator)
    return float(np.clip(cosine, 0.0, 1.0))
