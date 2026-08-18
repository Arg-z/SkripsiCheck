"""Central configuration for SkripsiCheck.

Environment variables use the ``SKRIPSICHECK_`` prefix. Values here are kept
dependency-free so core modules and tests can import them cheaply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_SEMANTIC_MODEL = PROJECT_ROOT / "deployment" / "model"


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(f"SKRIPSICHECK_{name}", default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(f"SKRIPSICHECK_{name}", default))


def _database_url_from_environment() -> str:
    """Resolve database configuration with Vercel/Neon-compatible fallbacks."""

    for variable in (
        "SKRIPSICHECK_DATABASE_URL",
        "DATABASE_URL",
        "POSTGRES_URL",
    ):
        value = os.getenv(variable)
        if value and value.strip():
            return value.strip()
    return "sqlite:///data/skripsicheck.sqlite3"


def _access_token_from_environment() -> str | None:
    value = os.getenv("SKRIPSICHECK_ACCESS_TOKEN")
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _semantic_model_path_from_environment() -> str | None:
    configured = os.getenv("SKRIPSICHECK_SEMANTIC_MODEL_PATH")
    if configured and configured.strip():
        path = Path(configured.strip())
        return str(path if path.is_absolute() else PROJECT_ROOT / path)
    if BUNDLED_SEMANTIC_MODEL.is_dir():
        return str(BUNDLED_SEMANTIC_MODEL)
    return None


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by all layers."""

    semantic_model: str = os.getenv(
        "SKRIPSICHECK_SEMANTIC_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    semantic_model_path: str | None = field(
        default_factory=_semantic_model_path_from_environment
    )
    lexical_weight: float = _env_float("LEXICAL_WEIGHT", 0.35)
    semantic_weight: float = _env_float("SEMANTIC_WEIGHT", 0.45)
    ngram_weight: float = _env_float("NGRAM_WEIGHT", 0.20)
    min_match_score: float = _env_float("MIN_MATCH_SCORE", 0.40)
    min_semantic_score: float = _env_float("MIN_SEMANTIC_SCORE", 0.40)
    top_k_matches: int = _env_int("TOP_K_MATCHES", 5)
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 25)
    max_analysis_characters: int = _env_int("MAX_ANALYSIS_CHARACTERS", 1_500_000)
    max_analysis_paragraphs: int = _env_int("MAX_ANALYSIS_PARAGRAPHS", 2_000)
    ngram_size: int = _env_int("NGRAM_SIZE", 5)
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 32)
    device: str = os.getenv("SKRIPSICHECK_DEVICE", "cpu")
    index_dir: Path = Path(os.getenv("SKRIPSICHECK_INDEX_DIR", "data/index"))
    upload_dir: Path = Path(os.getenv("SKRIPSICHECK_UPLOAD_DIR", "data/uploads"))
    storage_backend: str = os.getenv("SKRIPSICHECK_STORAGE_BACKEND", "local").lower()
    blob_document_prefix: str = os.getenv(
        "SKRIPSICHECK_BLOB_DOCUMENT_PREFIX", "documents"
    )
    index_backend: str = os.getenv("SKRIPSICHECK_INDEX_BACKEND", "local").lower()
    blob_index_prefix: str = os.getenv(
        "SKRIPSICHECK_BLOB_INDEX_PREFIX", "indexes/current"
    )
    access_token: str | None = field(
        default_factory=_access_token_from_environment,
        repr=False,
    )
    database_url: str = field(default_factory=_database_url_from_environment)

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
        if (
            self.max_upload_mb <= 0
            or self.max_analysis_characters <= 0
            or self.max_analysis_paragraphs <= 0
            or self.top_k_matches <= 0
            or self.embedding_batch_size <= 0
        ):
            raise ValueError("Limits and top-k values must be positive.")
        if not self.device.strip():
            raise ValueError("Embedding device cannot be empty.")
        if self.semantic_model_path is not None and not self.semantic_model_path.strip():
            raise ValueError("Semantic model path cannot be blank.")
        if not self.database_url.strip():
            raise ValueError("Database URL cannot be empty.")
        if self.storage_backend not in {"local", "vercel_blob"}:
            raise ValueError("Storage backend must be 'local' or 'vercel_blob'.")
        if self.index_backend not in {"local", "vercel_blob"}:
            raise ValueError("Index backend must be 'local' or 'vercel_blob'.")
        if not self.blob_document_prefix.strip() or not self.blob_index_prefix.strip():
            raise ValueError("Blob prefixes cannot be empty.")
        if self.access_token is not None and not 32 <= len(self.access_token) <= 512:
            raise ValueError("Access token must contain between 32 and 512 characters.")
        if self.storage_backend == "vercel_blob" and self.access_token is None:
            raise ValueError("Vercel Blob storage requires a shared access token.")
        if (
            self.storage_backend == "vercel_blob"
            and self.blob_document_prefix != "documents"
        ):
            raise ValueError(
                "Vercel Blob browser uploads currently require "
                "SKRIPSICHECK_BLOB_DOCUMENT_PREFIX=documents."
            )


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
MAX_ANALYSIS_CHARACTERS = SETTINGS.max_analysis_characters
MAX_ANALYSIS_PARAGRAPHS = SETTINGS.max_analysis_paragraphs
EMBEDDING_BATCH_SIZE = SETTINGS.embedding_batch_size
DEVICE = SETTINGS.device
INDEX_DIR = SETTINGS.index_dir
UPLOAD_DIR = SETTINGS.upload_dir
DATABASE_URL = SETTINGS.database_url
