"""Persistence boundary for documents and immutable analysis reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.database.entities import AnalysisEntity, DocumentEntity
from app.database.session import Database


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    original_filename: str
    stored_path: str
    media_type: str
    extension: str
    size_bytes: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    id: str
    document_id: str
    overall_similarity: float
    total_paragraphs: int
    matched_paragraphs: int
    result: dict[str, Any]
    created_at: datetime


def _document_record(entity: DocumentEntity) -> DocumentRecord:
    return DocumentRecord(
        id=entity.id,
        original_filename=entity.original_filename,
        stored_path=entity.stored_path,
        media_type=entity.media_type,
        extension=entity.extension,
        size_bytes=entity.size_bytes,
        created_at=entity.created_at,
    )


def _analysis_record(entity: AnalysisEntity) -> AnalysisRecord:
    result = json.loads(entity.result_json)
    if not isinstance(result, dict):
        raise ValueError("Stored analysis report is invalid.")
    return AnalysisRecord(
        id=entity.id,
        document_id=entity.document_id,
        overall_similarity=entity.overall_similarity,
        total_paragraphs=entity.total_paragraphs,
        matched_paragraphs=entity.matched_paragraphs,
        result=result,
        created_at=entity.created_at,
    )


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add_document(
        self,
        *,
        document_id: str,
        original_filename: str,
        stored_path: str,
        media_type: str,
        extension: str,
        size_bytes: int,
    ) -> DocumentRecord:
        with self.database.session() as session:
            entity = DocumentEntity(
                id=document_id,
                original_filename=original_filename,
                stored_path=stored_path,
                media_type=media_type,
                extension=extension,
                size_bytes=size_bytes,
            )
            session.add(entity)
            session.flush()
            return _document_record(entity)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self.database.session() as session:
            entity = session.get(DocumentEntity, document_id)
            return _document_record(entity) if entity is not None else None

    def delete_document(self, document_id: str) -> bool:
        with self.database.session() as session:
            entity = session.get(DocumentEntity, document_id)
            if entity is None:
                return False
            session.delete(entity)
            return True

    def add_analysis(
        self,
        *,
        analysis_id: str,
        document_id: str,
        overall_similarity: float,
        total_paragraphs: int,
        matched_paragraphs: int,
        result: dict[str, Any],
    ) -> AnalysisRecord:
        with self.database.session() as session:
            entity = AnalysisEntity(
                id=analysis_id,
                document_id=document_id,
                overall_similarity=overall_similarity,
                total_paragraphs=total_paragraphs,
                matched_paragraphs=matched_paragraphs,
                result_json=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            )
            session.add(entity)
            session.flush()
            return _analysis_record(entity)

    def get_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        with self.database.session() as session:
            entity = session.scalar(
                select(AnalysisEntity).where(AnalysisEntity.id == analysis_id)
            )
            return _analysis_record(entity) if entity is not None else None

