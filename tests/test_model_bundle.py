from __future__ import annotations

import tomllib
from pathlib import Path

from app.config import Settings
from app.core.indexer import SourceIndexer
from app.services.container import build_container
from app.services.vercel_index_cache import VercelBlobIndexRetriever
from scripts.prepare_vercel_model import MODEL_ID, MODEL_REVISION, REQUIRED_FILES


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_build_prepares_pinned_multilingual_model() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["vercel"]["scripts"]["build"] == (
        "python scripts/prepare_vercel_model.py"
    )
    assert MODEL_ID == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert len(MODEL_REVISION) == 40
    assert "model.safetensors" in REQUIRED_FILES


def test_model_weights_are_ignored_but_included_in_function_config() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    vercel = (ROOT / "vercel.json").read_text(encoding="utf-8")

    assert "deployment/model/" in ignored
    assert "deployment/model/**" in vercel


def test_container_passes_configured_model_source_to_local_indexer(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "deployment-model"
    settings = Settings(
        semantic_model="configured-semantic-model",
        semantic_model_path=str(model_path),
        embedding_batch_size=7,
        device="cpu",
        index_dir=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
    )
    container = build_container(
        settings=settings,
        database_url=f"sqlite:///{(tmp_path / 'app.sqlite3').as_posix()}",
    )
    try:
        retriever = container.analysis_service.retriever_factory()
        assert isinstance(retriever, SourceIndexer)
        semantic = retriever.semantic_service
        assert semantic.model_name == "configured-semantic-model"
        assert semantic.model_source == str(model_path)
        assert semantic.batch_size == 7
        assert semantic.device == "cpu"
    finally:
        container.database.close()


def test_container_passes_configured_model_source_to_blob_index_cache(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "deployment-model"
    settings = Settings(
        semantic_model="configured-semantic-model",
        semantic_model_path=str(model_path),
        index_backend="vercel_blob",
        blob_index_prefix="indexes/test-generation",
        upload_dir=tmp_path / "uploads",
    )
    container = build_container(
        settings=settings,
        database_url=f"sqlite:///{(tmp_path / 'app.sqlite3').as_posix()}",
    )
    try:
        retriever = container.analysis_service.retriever_factory()
        assert isinstance(retriever, VercelBlobIndexRetriever)
        assert container.index_cache is not None
        semantic = container.index_cache.semantic_service
        assert semantic is not None
        assert semantic.model_name == "configured-semantic-model"
        assert semantic.model_source == str(model_path)
    finally:
        container.database.close()
