"""Application service container shared by API dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import SETTINGS, Settings
from app.core.indexer import SourceIndexer
from app.database.repository import Repository
from app.database.session import Database
from app.services.analysis_service import AnalysisService
from app.services.document_storage import LocalDocumentStorage
from app.services.similarity_engine import CandidateRetriever


@dataclass(slots=True)
class AppContainer:
    database: Database
    repository: Repository
    storage: LocalDocumentStorage
    analysis_service: AnalysisService


def build_container(
    *,
    settings: Settings = SETTINGS,
    database_url: str | None = None,
    upload_dir: str | Path | None = None,
    retriever_factory: Callable[[], CandidateRetriever] | None = None,
) -> AppContainer:
    database = Database(database_url or settings.database_url)
    database.create_schema()
    repository = Repository(database)
    storage = LocalDocumentStorage(
        upload_dir or settings.upload_dir,
        max_upload_mb=settings.max_upload_mb,
    )
    factory = retriever_factory or (lambda: SourceIndexer(index_dir=settings.index_dir))
    return AppContainer(
        database=database,
        repository=repository,
        storage=storage,
        analysis_service=AnalysisService(repository, factory),
    )

