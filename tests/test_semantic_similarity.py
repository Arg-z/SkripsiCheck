from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pytest

from app.core.semantic_similarity import (
    SemanticSimilarityService,
    semantic_similarity,
)


class FakeSentenceTransformer:
    """Tiny deterministic encoder used to keep unit tests offline."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
        self.encode_calls += 1
        vectors: list[list[float]] = []
        for sentence in sentences:
            lowered = sentence.casefold()
            if "kalsium" in lowered or "calcium" in lowered:
                vectors.append([3.0, 0.2, 0.0])
            elif "enkripsi" in lowered or "encryption" in lowered:
                vectors.append([0.0, 3.0, 0.2])
            else:
                vectors.append([0.0, 0.0, 2.0])
        return np.asarray(vectors, dtype=np.float64)


def test_model_service_encodes_batch_and_loads_only_once() -> None:
    fake_model = FakeSentenceTransformer()
    factory_calls: list[tuple[str, str]] = []

    def factory(model_name: str, device: str) -> FakeSentenceTransformer:
        factory_calls.append((model_name, device))
        return fake_model

    service = SemanticSimilarityService(
        model_name="local-test-model",
        batch_size=2,
        device="cpu",
        model_factory=factory,
    )
    assert not service.is_loaded

    embeddings = service.encode_texts(["Kalsium untuk telur", "Keamanan enkripsi"])
    second_embedding = service.encode_text("Calcium affects eggshells")

    assert service.is_loaded
    assert factory_calls == [("local-test-model", "cpu")]
    assert fake_model.encode_calls == 2
    assert embeddings.shape == (2, 3)
    assert second_embedding.shape == (3,)
    assert embeddings.dtype == np.float32
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1.0, 1.0])


def test_empty_inputs_are_safe_and_preserve_batch_positions() -> None:
    fake_model = FakeSentenceTransformer()
    service = SemanticSimilarityService(model_factory=lambda _name, _device: fake_model)

    assert service.encode_texts([]).shape == (0, 0)
    assert not service.is_loaded

    embeddings = service.encode_texts(["", "Kalsium", "   "])
    assert embeddings.shape == (3, 3)
    assert np.count_nonzero(embeddings[0]) == 0
    assert np.count_nonzero(embeddings[2]) == 0
    assert np.linalg.norm(embeddings[1]) == pytest.approx(1.0)


def test_identical_text_has_high_semantic_similarity() -> None:
    service = SemanticSimilarityService(
        model_factory=lambda _name, _device: FakeSentenceTransformer()
    )
    first, second = service.encode_texts(
        ["Kalsium memengaruhi kualitas telur.", "Kalsium memengaruhi kualitas telur."]
    )
    assert semantic_similarity(first, second) == pytest.approx(1.0)


def test_related_text_scores_above_unrelated_text() -> None:
    service = SemanticSimilarityService(
        model_factory=lambda _name, _device: FakeSentenceTransformer()
    )
    query, related, unrelated = service.encode_texts(
        [
            "Kalsium memengaruhi kualitas cangkang telur.",
            "Calcium affects eggshell quality.",
            "Algoritma enkripsi melindungi jaringan.",
        ]
    )

    assert semantic_similarity(query, related) > 0.95
    assert semantic_similarity(query, unrelated) < 0.10


def test_similarity_clips_negative_cosine_and_handles_zero_vectors() -> None:
    assert semantic_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert semantic_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="same dimension"):
        semantic_similarity([1.0], [1.0, 0.0])


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("SKRIPSICHECK_RUN_SLOW") != "1",
    reason="Set SKRIPSICHECK_RUN_SLOW=1 after caching the model locally.",
)
def test_real_multilingual_model_semantics() -> None:
    """Optional local integration check; disabled by default to avoid downloads."""

    service = SemanticSimilarityService(device="cpu")
    query, paraphrase, unrelated = service.encode_texts(
        [
            "Kalsium meningkatkan kekuatan cangkang telur.",
            "Calcium makes eggshells stronger.",
            "Kriptografi digunakan untuk keamanan jaringan komputer.",
        ]
    )

    assert query.shape[0] > 0
    assert semantic_similarity(query, paraphrase) > semantic_similarity(query, unrelated)
