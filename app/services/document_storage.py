"""Safe local upload storage behind a replaceable service boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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


@dataclass(frozen=True, slots=True)
class StoredDocument:
    document_id: str
    original_filename: str
    stored_path: Path
    media_type: str
    extension: str
    size_bytes: int


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

