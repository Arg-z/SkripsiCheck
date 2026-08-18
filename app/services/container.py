"""Application service container shared by API dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import SETTINGS, Settings
from app.core.indexer import SourceIndexer
from app.core.semantic_similarity import SemanticSimilarityService
from app.database.repository import Repository
from app.database.session import Database
from app.services.analysis_service import AnalysisService
from app.services.document_storage import DocumentStorage, LocalDocumentStorage
from app.services.similarity_engine import CandidateRetriever
from app.services.vercel_blob_storage import VercelBlobDocumentStorage
from app.services.vercel_index_cache import (
    VercelBlobIndexCache,
    VercelBlobIndexRetriever,
)


@dataclass(slots=True)
class AppContainer:
    database: Database
    repository: Repository
    storage: DocumentStorage
    analysis_service: AnalysisService
    index_cache: VercelBlobIndexCache | None = None


def build_container(
    *,
    settings: Settings = SETTINGS,
    database_url: str | None = None,
    upload_dir: str | Path | None = None,
    retriever_factory: Callable[[], CandidateRetriever] | None = None,
) -> AppContainer:
    settings.validate()
    database = Database(database_url or settings.database_url)
    database.create_schema()
    repository = Repository(database)
    if settings.storage_backend == "vercel_blob":
        if upload_dir is not None:
            raise ValueError("upload_dir cannot be used with Vercel Blob storage.")
        storage: DocumentStorage = VercelBlobDocumentStorage(
            max_upload_mb=settings.max_upload_mb,
            prefix=settings.blob_document_prefix,
        )
    else:
        storage = LocalDocumentStorage(
            upload_dir or settings.upload_dir,
            max_upload_mb=settings.max_upload_mb,
        )
    index_cache: VercelBlobIndexCache | None = None
    factory = retriever_factory
    if factory is None:
        semantic_service = SemanticSimilarityService(
            model_name=settings.semantic_model,
            model_path=settings.semantic_model_path,
            batch_size=settings.embedding_batch_size,
            device=settings.device,
        )
        if settings.index_backend == "vercel_blob":
            index_cache = VercelBlobIndexCache(
                prefix=settings.blob_index_prefix,
                semantic_service=semantic_service,
            )
            factory = lambda: VercelBlobIndexRetriever(index_cache)
        else:
            factory = lambda: SourceIndexer(
                semantic_service,
                index_dir=settings.index_dir,
            )
    return AppContainer(
        database=database,
        repository=repository,
        storage=storage,
        analysis_service=AnalysisService(
            repository,
            factory,
            storage,
            max_characters=settings.max_analysis_characters,
            max_paragraphs=settings.max_analysis_paragraphs,
        ),
        index_cache=index_cache,
    )
