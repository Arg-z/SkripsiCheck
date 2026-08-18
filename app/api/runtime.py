"""Public, secret-free runtime capabilities used by the browser client."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import RuntimeResponse


router = APIRouter(prefix="/api", tags=["system"])


@router.get("/runtime", response_model=RuntimeResponse)
def get_runtime(request: Request) -> RuntimeResponse:
    """Describe browser requirements without disclosing configured secrets."""

    return RuntimeResponse(
        access_required=bool(request.app.state.access_required),
        direct_upload=bool(request.app.state.direct_upload),
        max_upload_mb=int(request.app.state.max_upload_mb),
    )
