"""Document upload, metadata, and deletion endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import get_container, get_owner_session_id
from app.database.repository import DocumentRecord
from app.models.schemas import BlobDocumentFinalizeRequest, DocumentResponse
from app.services.container import AppContainer
from app.services.document_storage import (
    UploadDocumentStorage,
    StorageUnavailableError,
    UploadStorageError,
    UploadTooLargeError,
)
from app.services.vercel_blob_storage import StoredBlobDocument, VercelBlobDocumentStorage

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


def _same_blob_document(record: DocumentRecord, stored: StoredBlobDocument) -> bool:
    """Return whether a finalized Blob is the already-persisted same document."""

    return (
        record.id == stored.document_id
        and record.original_filename == stored.original_filename
        and record.stored_path == stored.stored_path
        and record.media_type == stored.media_type
        and record.extension == stored.extension
        and record.size_bytes == stored.size_bytes
    )


def _delete_blob_quietly(storage: VercelBlobDocumentStorage, pathname: str) -> None:
    """Best-effort cleanup without hiding the primary database failure."""

    try:
        storage.delete(pathname)
    except UploadStorageError:
        return


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, or TXT document")],
    container: Annotated[AppContainer, Depends(get_container)],
    owner_session_id: Annotated[str, Depends(get_owner_session_id)],
) -> DocumentResponse:
    if not isinstance(container.storage, UploadDocumentStorage):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This deployment requires a direct private Blob upload. "
                "Use the SkripsiCheck web interface."
            ),
        )
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
            owner_session_id=owner_session_id,
        )
    except Exception:
        container.storage.delete(stored.stored_path)
        raise
    return _response(record)


@router.post(
    "/blob",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def finalize_blob_document(
    request: BlobDocumentFinalizeRequest,
    container: Annotated[AppContainer, Depends(get_container)],
    owner_session_id: Annotated[str, Depends(get_owner_session_id)],
) -> DocumentResponse:
    """Verify a completed direct browser upload and persist its metadata."""

    if not isinstance(container.storage, VercelBlobDocumentStorage):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Direct Blob upload is not enabled for this deployment.",
        )
    try:
        stored = container.storage.finalize_browser_upload(
            request.pathname,
            request.filename,
            session_id=owner_session_id,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
        ) from exc
    except StorageUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except UploadStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    # A browser can legitimately retry this request after the database commit
    # if its first response was lost. Return the existing record instead of
    # attempting another INSERT and deleting the valid Blob during cleanup.
    existing = container.repository.get_document(stored.document_id, owner_session_id)
    if existing is not None:
        if _same_blob_document(existing, stored):
            return _response(existing)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document identifier is already in use.",
        )

    try:
        record = container.repository.add_document(
            document_id=stored.document_id,
            original_filename=stored.original_filename,
            stored_path=stored.stored_path,
            media_type=stored.media_type,
            extension=stored.extension,
            size_bytes=stored.size_bytes,
            owner_session_id=owner_session_id,
        )
    except IntegrityError as exc:
        # Handle a concurrent retry that won the INSERT race.
        existing = container.repository.get_document(
            stored.document_id, owner_session_id
        )
        if existing is not None and _same_blob_document(existing, stored):
            return _response(existing)
        _delete_blob_quietly(container.storage, stored.stored_path)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document identifier is already in use.",
        ) from exc
    except Exception:
        _delete_blob_quietly(container.storage, stored.stored_path)
        raise
    return _response(record)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    container: Annotated[AppContainer, Depends(get_container)],
    owner_session_id: Annotated[str, Depends(get_owner_session_id)],
) -> DocumentResponse:
    record = container.repository.get_document(str(document_id), owner_session_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return _response(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    container: Annotated[AppContainer, Depends(get_container)],
    owner_session_id: Annotated[str, Depends(get_owner_session_id)],
) -> None:
    record = container.repository.get_document(str(document_id), owner_session_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    try:
        container.storage.delete(record.stored_path)
    except StorageUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    container.repository.delete_document(record.id, owner_session_id)
