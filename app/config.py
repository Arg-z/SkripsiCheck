"""Central configuration for SkripsiCheck.

Environment variables use the ``SKRIPSICHECK_`` prefix. Values here are kept
dependency-free so core modules and tests can import them cheaply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(f"SKRIPSICHECK_{name}", default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(f"SKRIPSICHECK_{name}", default))


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by all layers."""

    semantic_model: str = os.getenv(
        "SKRIPSICHECK_SEMANTIC_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    lexical_weight: float = _env_float("LEXICAL_WEIGHT", 0.35)
    semantic_weight: float = _env_float("SEMANTIC_WEIGHT", 0.45)
    ngram_weight: float = _env_float("NGRAM_WEIGHT", 0.20)
    min_match_score: float = _env_float("MIN_MATCH_SCORE", 0.40)
    min_semantic_score: float = _env_float("MIN_SEMANTIC_SCORE", 0.40)
    top_k_matches: int = _env_int("TOP_K_MATCHES", 5)
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 25)
    ngram_size: int = _env_int("NGRAM_SIZE", 5)
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 32)
    device: str = os.getenv("SKRIPSICHECK_DEVICE", "cpu")
    index_dir: Path = Path(os.getenv("SKRIPSICHECK_INDEX_DIR", "data/index"))
    upload_dir: Path = Path(os.getenv("SKRIPSICHECK_UPLOAD_DIR", "data/uploads"))
    database_url: str = os.getenv(
        "SKRIPSICHECK_DATABASE_URL", "sqlite:///data/skripsicheck.sqlite3"
    )

    def validate(self) -> None:
        weights = (self.lexical_weight, self.semantic_weight, self.ngram_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("Similarity weights cannot be negative.")
        if not 0.999 <= sum(weights) <= 1.001:
            raise ValueError("Similarity weights must sum to 1.0.")
        if not 0.0 <= self.min_match_score <= 1.0:
            raise ValueError("Minimum match score must be in the range 0..1.")
        if not 0.0 <= self.min_semantic_score <= 1.0:
            raise ValueError("Minimum semantic score must be in the range 0..1.")
        if self.max_upload_mb <= 0 or self.top_k_matches <= 0 or self.embedding_batch_size <= 0:
            raise ValueError("Limits and top-k values must be positive.")
        if not self.device.strip():
            raise ValueError("Embedding device cannot be empty.")
        if not self.database_url.strip():
            raise ValueError("Database URL cannot be empty.")


SETTINGS = Settings()
SETTINGS.validate()

# Convenient named constants for integrations and documentation.
SEMANTIC_MODEL = SETTINGS.semantic_model
LEXICAL_WEIGHT = SETTINGS.lexical_weight
SEMANTIC_WEIGHT = SETTINGS.semantic_weight
NGRAM_WEIGHT = SETTINGS.ngram_weight
MIN_MATCH_SCORE = SETTINGS.min_match_score
MIN_SEMANTIC_SCORE = SETTINGS.min_semantic_score
TOP_K_MATCHES = SETTINGS.top_k_matches
MAX_UPLOAD_MB = SETTINGS.max_upload_mb
EMBEDDING_BATCH_SIZE = SETTINGS.embedding_batch_size
DEVICE = SETTINGS.device
INDEX_DIR = SETTINGS.index_dir
UPLOAD_DIR = SETTINGS.upload_dir
DATABASE_URL = SETTINGS.database_url
