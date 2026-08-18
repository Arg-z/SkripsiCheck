"""Persisted JSON similarity report endpoint."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container, get_owner_session_id
from app.models.schemas import ReportResponse
from app.services.container import AppContainer

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{analysis_id}", response_model=ReportResponse)
def get_report(
    analysis_id: UUID,
    container: Annotated[AppContainer, Depends(get_container)],
    owner_session_id: Annotated[str, Depends(get_owner_session_id)],
) -> ReportResponse:
    record = container.repository.get_analysis(str(analysis_id), owner_session_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    payload = dict(record.result)
    payload["created_at"] = record.created_at
    return ReportResponse.model_validate(payload)
