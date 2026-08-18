"""Synchronous MVP document analysis endpoint."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container, get_owner_session_id
from app.core.indexer import IndexIntegrityError, NoSimilarityIndexError, SourceIndexError
from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import (
    AnalysisLimitExceededError,
    AnalysisStorageUnavailableError,
    AnalysisServiceError,
    DocumentNotFoundError,
    EmptyDocumentError,
)
from app.services.container import AppContainer

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze_document(
    request: AnalysisRequest,
    container: Annotated[AppContainer, Depends(get_container)],
    owner_session_id: Annotated[str, Depends(get_owner_session_id)],
) -> AnalysisResponse:
    try:
        record = container.analysis_service.analyze(
            str(request.document_id),
            owner_session_id=owner_session_id,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except AnalysisLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
        ) from exc
    except (NoSimilarityIndexError, IndexIntegrityError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except SourceIndexError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except AnalysisStorageUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except AnalysisServiceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    return AnalysisResponse(
        id=UUID(record.id),
        document_id=UUID(record.document_id),
        overall_similarity=record.overall_similarity,
        total_paragraphs=record.total_paragraphs,
        matched_paragraphs=record.matched_paragraphs,
        created_at=record.created_at,
        report_url=f"/api/reports/{record.id}",
    )
