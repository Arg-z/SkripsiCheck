"""Pydantic request and response schemas for PHASE 4."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    phase: int


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    media_type: str
    extension: str
    size_bytes: int
    created_at: datetime


class AnalysisRequest(BaseModel):
    document_id: UUID
    top_k: int | None = Field(default=None, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    id: UUID
    document_id: UUID
    overall_similarity: float = Field(ge=0.0, le=1.0)
    total_paragraphs: int = Field(ge=0)
    matched_paragraphs: int = Field(ge=0)
    created_at: datetime
    report_url: str


class MatchResponse(BaseModel):
    chunk_id: str
    source_file: str
    source_path: str | None = None
    matched_text: str
    lexical_similarity: float
    semantic_similarity: float
    ngram_overlap: float
    final_score: float
    risk: str
    reason: str
    word_count: int
    page: int | None = None


class ParagraphResponse(BaseModel):
    number: int
    text: str
    word_count: int
    candidates_retrieved: int
    matches: list[MatchResponse]


class ReportDocumentResponse(BaseModel):
    id: UUID
    filename: str


class ReportResponse(BaseModel):
    analysis_id: UUID
    document: ReportDocumentResponse
    overall_similarity: float
    total_paragraphs: int
    matched_paragraphs: int
    paragraphs: list[ParagraphResponse]
    methodology: dict[str, Any]
    disclaimer: str
    created_at: datetime

