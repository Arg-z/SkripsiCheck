"""Document upload, metadata, and deletion endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_container
from app.database.repository import DocumentRecord
from app.models.schemas import DocumentResponse
from app.services.container import AppContainer
from app.services.document_storage import UploadStorageError, UploadTooLargeError

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _response(record: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=UUID(record.id),
        filename=record.original_filename,
        media_type=record.media_type,
        extension=record.extension,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, or TXT document")],
    container: Annotated[AppContainer, Depends(get_container)],
) -> DocumentResponse:
    try:
        stored = await container.storage.save(file)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except UploadStorageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    try:
        record = container.repository.add_document(
            document_id=stored.document_id,
            original_filename=stored.original_filename,
            stored_path=str(stored.stored_path),
            media_type=stored.media_type,
            extension=stored.extension,
            size_bytes=stored.size_bytes,
        )
    except Exception:
        container.storage.delete(stored.stored_path)
        raise
    return _response(record)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    container: Annotated[AppContainer, Depends(get_container)],
) -> DocumentResponse:
    record = container.repository.get_document(str(document_id))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return _response(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    container: Annotated[AppContainer, Depends(get_container)],
) -> None:
    record = container.repository.get_document(str(document_id))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    container.storage.delete(record.stored_path)
    container.repository.delete_document(record.id)
