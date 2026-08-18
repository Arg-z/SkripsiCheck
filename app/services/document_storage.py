"""Safe local upload storage behind a replaceable service boundary."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable
from uuid import uuid4

from fastapi import UploadFile

from app.core.extractor import DocumentExtractionError, extract_text

ALLOWED_MEDIA_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    ".txt": frozenset({"text/plain"}),
}
SAFE_NAME_RE = re.compile(r"[^\w.()\- ]+", flags=re.UNICODE)


class UploadStorageError(ValueError):
    """Base class for a rejected or failed upload."""


class UploadTooLargeError(UploadStorageError):
    """Raised as soon as a streaming upload exceeds the configured limit."""


class StorageUnavailableError(UploadStorageError):
    """Raised when a configured storage service cannot complete an operation."""


@dataclass(frozen=True, slots=True)
class StoredDocument:
    document_id: str
    original_filename: str
    stored_path: Path
    media_type: str
    extension: str
    size_bytes: int


class DocumentStorage(Protocol):
    """Storage operations required after a document has been persisted."""

    def delete(self, stored_path: str | Path) -> bool: ...

    def materialize_for_analysis(
        self,
        stored_path: str | Path,
        *,
        expected_extension: str | None = None,
        expected_media_type: str | None = None,
        expected_size: int | None = None,
        validate_document: bool = False,
    ) -> Iterator[Path]: ...

    def close(self) -> None: ...


@runtime_checkable
class UploadDocumentStorage(DocumentStorage, Protocol):
    """Storage backend that accepts multipart uploads through FastAPI."""

    async def save(self, upload: UploadFile) -> StoredDocument: ...


def sanitize_filename(filename: str | None) -> str:
    """Remove path components and unsafe characters from a display filename."""

    if not filename:
        raise UploadStorageError("A filename is required.")
    basename = PurePosixPath(filename.replace("\\", "/")).name.strip()
    safe = SAFE_NAME_RE.sub("_", basename).strip(" .")
    if not safe or safe in {".", ".."}:
        raise UploadStorageError("The filename is invalid.")
    return safe[:255]


class LocalDocumentStorage:
    """Write generated filenames under one local directory; never trust user paths."""

    def __init__(self, root: str | Path, *, max_upload_mb: int) -> None:
        self.root = Path(root)
        self.max_bytes = max_upload_mb * 1024 * 1024
        if self.max_bytes <= 0:
            raise ValueError("max_upload_mb must be positive.")
        self.root.mkdir(parents=True, exist_ok=True)

    async def save(self, upload: UploadFile) -> StoredDocument:
        filename = sanitize_filename(upload.filename)
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_MEDIA_TYPES:
            raise UploadStorageError("Only PDF, DOCX, and TXT files are supported.")
        media_type = (upload.content_type or "").split(";", maxsplit=1)[0].strip().lower()
        if media_type not in ALLOWED_MEDIA_TYPES[extension]:
            expected = ", ".join(sorted(ALLOWED_MEDIA_TYPES[extension]))
            raise UploadStorageError(
                f"MIME type {media_type or '(missing)'} is invalid for {extension}; expected {expected}."
            )

        document_id = str(uuid4())
        target = self.root / f"{document_id}{extension}"
        size = 0
        try:
            with target.open("xb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise UploadTooLargeError(
                            f"File exceeds the {self.max_bytes // (1024 * 1024)} MB limit."
                        )
                    destination.write(chunk)
            if size == 0:
                raise UploadStorageError("The uploaded file is empty.")
            # Validate actual container/signature and readability after streaming.
            extract_text(target, max_size_mb=self.max_bytes // (1024 * 1024))
        except (DocumentExtractionError, OSError) as exc:
            target.unlink(missing_ok=True)
            raise UploadStorageError(f"The uploaded document is invalid: {exc}") from exc
        except UploadStorageError:
            target.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        return StoredDocument(
            document_id=document_id,
            original_filename=filename,
            stored_path=target.resolve(),
            media_type=media_type,
            extension=extension,
            size_bytes=size,
        )

    def delete(self, stored_path: str | Path) -> bool:
        candidate = Path(stored_path).resolve()
        root = self.root.resolve()
        if candidate.parent != root:
            raise UploadStorageError("Refusing to delete a file outside upload storage.")
        if not candidate.exists():
            return False
        candidate.unlink()
        return True

    @contextmanager
    def materialize_for_analysis(
        self,
        stored_path: str | Path,
        *,
        expected_extension: str | None = None,
        expected_media_type: str | None = None,
        expected_size: int | None = None,
        validate_document: bool = False,
    ) -> Iterator[Path]:
        """Yield a validated local upload through the shared storage boundary."""

        del expected_media_type
        candidate = Path(stored_path).resolve()
        root = self.root.resolve()
        if candidate.parent != root:
            raise UploadStorageError("Refusing to read a file outside upload storage.")
        if not candidate.is_file():
            raise UploadStorageError("The stored document no longer exists.")
        if expected_extension is not None and candidate.suffix.lower() != expected_extension:
            raise UploadStorageError("Stored document extension does not match its metadata.")
        if expected_size is not None and candidate.stat().st_size != expected_size:
            raise UploadStorageError("Stored document size does not match its metadata.")
        if validate_document:
            try:
                extract_text(candidate, max_size_mb=self.max_bytes // (1024 * 1024))
            except DocumentExtractionError as exc:
                raise UploadStorageError(
                    f"The stored document is invalid: {exc}"
                ) from exc
        yield candidate

    def close(self) -> None:
        """Local storage owns no persistent transport."""

        return None
