"""Private Vercel Blob storage boundary for serverless document analysis.

The browser-upload token exchange is intentionally not implemented here.  Vercel's
Python SDK (``vercel`` 0.7.x) provides Blob object operations, while the official
client-upload token handler is currently part of ``@vercel/blob/client``.  A route
using that handler should apply every field from :class:`DirectUploadPlan` before
the browser uploads directly to Blob.  This module then verifies the uploaded
object, materializes it under the platform temporary directory for extraction,
and deletes it when requested.

No read-write token is returned, logged, or included in an exception message.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from uuid import RFC_4122, UUID, uuid4

from app.core.extractor import DocumentExtractionError, extract_text
from app.services.document_storage import (
    ALLOWED_MEDIA_TYPES,
    StorageUnavailableError,
    UploadStorageError,
    UploadTooLargeError,
    sanitize_filename,
)

MEBIBYTE = 1024 * 1024
DEFAULT_TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"
DEFAULT_BLOB_PREFIX = "documents"


class BlobStorageError(StorageUnavailableError):
    """Base class for private Blob configuration or operation failures."""


class BlobStorageConfigurationError(BlobStorageError):
    """Raised when the Vercel Blob SDK or its credential is unavailable."""


class BlobUploadValidationError(UploadStorageError):
    """Raised when a completed direct upload does not match its server plan."""


class BlobClientProtocol(Protocol):
    """Subset of ``vercel.blob.BlobClient`` used by this adapter."""

    def head(self, url_or_path: str, *, token: str | None = None) -> Any: ...

    def download_file(
        self,
        url_or_path: str,
        local_path: str | os.PathLike[str],
        *,
        access: str = "public",
        timeout: float | None = None,
        overwrite: bool = True,
        create_parents: bool = True,
        token: str | None = None,
    ) -> str: ...

    def delete(
        self, url_or_path: str, *, token: str | None = None
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DirectUploadPlan:
    """Server-generated constraints consumed by the client-upload token route."""

    document_id: str
    original_filename: str
    pathname: str
    media_type: str
    extension: str
    declared_size_bytes: int
    maximum_size_bytes: int
    multipart: bool = True


@dataclass(frozen=True, slots=True)
class StoredBlobDocument:
    """Verified metadata suitable for persistence after a direct upload."""

    document_id: str
    original_filename: str
    stored_path: str
    blob_url: str
    media_type: str
    extension: str
    size_bytes: int
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class _BlobMetadata:
    pathname: str
    url: str
    content_type: str
    size: int
    etag: str | None


def _default_client_factory(token: str) -> BlobClientProtocol:
    try:
        from vercel.blob import BlobClient
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise BlobStorageConfigurationError(
            "Vercel Blob support requires the 'vercel' Python package."
        ) from exc
    return BlobClient(token=token)


class VercelBlobDocumentStorage:
    """Manage sensitive documents in a private Vercel Blob namespace.

    ``prepare_direct_upload`` performs all checks possible before issuing a
    browser-upload token. ``finalize_direct_upload`` distrusts client-provided
    metadata: it reads metadata from Blob and validates the actual downloaded
    document before returning a persistence record.
    """

    def __init__(
        self,
        *,
        max_upload_mb: int,
        prefix: str = DEFAULT_BLOB_PREFIX,
        temp_root: str | Path | None = None,
        token_env: str = DEFAULT_TOKEN_ENV,
        client: BlobClientProtocol | None = None,
        client_factory: Callable[[str], BlobClientProtocol] = _default_client_factory,
    ) -> None:
        if max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive.")
        normalized_prefix = str(PurePosixPath(prefix.strip("/")))
        if (
            not normalized_prefix
            or normalized_prefix in {".", ".."}
            or ".." in PurePosixPath(normalized_prefix).parts
        ):
            raise ValueError("Blob prefix must be a safe relative path.")
        if not token_env.strip():
            raise ValueError("token_env cannot be empty.")

        self.max_bytes = max_upload_mb * MEBIBYTE
        self.prefix = normalized_prefix
        self.temp_root = Path(temp_root or tempfile.gettempdir())
        self.token_env = token_env
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None

    def prepare_direct_upload(
        self,
        filename: str | None,
        media_type: str | None,
        size_bytes: int,
    ) -> DirectUploadPlan:
        """Validate browser metadata and reserve an unguessable Blob pathname."""

        safe_name = sanitize_filename(filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in ALLOWED_MEDIA_TYPES:
            raise UploadStorageError("Only PDF, DOCX, and TXT files are supported.")
        normalized_media_type = (media_type or "").split(";", maxsplit=1)[0].strip().lower()
        if normalized_media_type not in ALLOWED_MEDIA_TYPES[extension]:
            expected = ", ".join(sorted(ALLOWED_MEDIA_TYPES[extension]))
            raise UploadStorageError(
                f"MIME type {normalized_media_type or '(missing)'} is invalid for "
                f"{extension}; expected {expected}."
            )
        if size_bytes <= 0:
            raise UploadStorageError("The uploaded file is empty.")
        if size_bytes > self.max_bytes:
            raise UploadTooLargeError(
                f"File exceeds the {self.max_bytes // MEBIBYTE} MB limit."
            )

        document_id = str(uuid4())
        return DirectUploadPlan(
            document_id=document_id,
            original_filename=safe_name,
            pathname=f"{self.prefix}/{document_id}{extension}",
            media_type=normalized_media_type,
            extension=extension,
            declared_size_bytes=size_bytes,
            maximum_size_bytes=self.max_bytes,
        )

    def finalize_direct_upload(self, plan: DirectUploadPlan) -> StoredBlobDocument:
        """Verify Blob metadata and file contents after a direct browser upload.

        The caller must retrieve ``plan`` from trusted server-side state or a
        verified token payload. It must never rebuild the plan solely from fields
        supplied by the browser.
        """

        self._validate_pathname(plan.pathname)
        metadata = self._head(plan.pathname)
        try:
            self._validate_completed_upload(plan, metadata)
            with self.materialize_for_analysis(
                plan.pathname,
                expected_extension=plan.extension,
                expected_media_type=plan.media_type,
                expected_size=metadata.size,
                validate_document=True,
            ):
                pass
        except BlobStorageError:
            # A transient SDK/network/temp-storage failure says nothing about
            # the validity of the uploaded object. Keep it so the client can
            # safely retry finalization.
            raise
        except UploadStorageError:
            # A rejected upload must not leave user-controlled data in storage.
            self._delete_quietly(plan.pathname)
            raise

        return StoredBlobDocument(
            document_id=plan.document_id,
            original_filename=plan.original_filename,
            stored_path=metadata.pathname,
            blob_url=metadata.url,
            media_type=metadata.content_type,
            extension=plan.extension,
            size_bytes=metadata.size,
            etag=metadata.etag,
        )

    def finalize_browser_upload(
        self,
        pathname: str,
        original_filename: str | None,
        *,
        session_id: str,
    ) -> StoredBlobDocument:
        """Verify an untrusted browser upload without accepting client metadata.

        The JavaScript token route constrains the object before upload. This
        method independently verifies its session namespace, UUID filename,
        private Blob metadata, size, MIME type, and actual document contents.
        """

        safe_name = sanitize_filename(original_filename)
        candidate = PurePosixPath(pathname)
        expected_parent = PurePosixPath(self.prefix) / session_id
        if candidate.parent != expected_parent:
            raise BlobUploadValidationError(
                "Blob pathname does not belong to this browser session."
            )
        extension = candidate.suffix.lower()
        if extension not in ALLOWED_MEDIA_TYPES or Path(safe_name).suffix.lower() != extension:
            raise BlobUploadValidationError(
                "Blob extension does not match the original filename."
            )
        try:
            parsed_document_id = UUID(candidate.stem)
            parsed_session_id = UUID(session_id)
        except ValueError:
            raise BlobUploadValidationError(
                "Blob pathname contains an invalid identifier."
            ) from None
        if (
            parsed_document_id.version != 4
            or parsed_document_id.variant != RFC_4122
            or parsed_session_id.version != 4
            or parsed_session_id.variant != RFC_4122
            or candidate.stem != str(parsed_document_id)
            or session_id != str(parsed_session_id)
        ):
            raise BlobUploadValidationError(
                "Blob pathname identifiers are not canonical UUIDv4 values."
            )
        document_id = str(parsed_document_id)

        self._validate_pathname(pathname)
        metadata = self._head(pathname)
        try:
            if metadata.pathname != pathname:
                raise BlobUploadValidationError(
                    "Blob pathname does not match its private metadata."
                )
            if metadata.content_type not in ALLOWED_MEDIA_TYPES[extension]:
                raise BlobUploadValidationError(
                    "Blob content type does not match its extension."
                )
            if metadata.size <= 0:
                raise BlobUploadValidationError("The uploaded file is empty.")
            if metadata.size > self.max_bytes:
                raise UploadTooLargeError(
                    f"File exceeds the {self.max_bytes // MEBIBYTE} MB limit."
                )
            with self.materialize_for_analysis(
                pathname,
                expected_extension=extension,
                expected_media_type=metadata.content_type,
                expected_size=metadata.size,
                validate_document=True,
            ):
                pass
        except BlobStorageError:
            raise
        except UploadStorageError:
            self._delete_quietly(pathname)
            raise

        return StoredBlobDocument(
            document_id=document_id,
            original_filename=safe_name,
            stored_path=metadata.pathname,
            blob_url=metadata.url,
            media_type=metadata.content_type,
            extension=extension,
            size_bytes=metadata.size,
            etag=metadata.etag,
        )

    @contextmanager
    def materialize_for_analysis(
        self,
        pathname: str,
        *,
        expected_extension: str | None = None,
        expected_media_type: str | None = None,
        expected_size: int | None = None,
        validate_document: bool = False,
    ) -> Iterator[Path]:
        """Download one private blob into a cleaned-up temporary directory.

        The yielded path remains valid only inside the context manager.  This is
        designed for Vercel's writable ``/tmp`` filesystem and prevents documents
        from leaking between analyses in a warm Function instance.
        """

        self._validate_pathname(pathname)
        metadata = self._head(pathname)
        extension = PurePosixPath(metadata.pathname).suffix.lower()
        if extension not in ALLOWED_MEDIA_TYPES:
            raise BlobUploadValidationError("Blob has an unsupported document extension.")
        if expected_extension is not None and extension != expected_extension.lower():
            raise BlobUploadValidationError("Blob extension does not match the upload plan.")
        if metadata.content_type not in ALLOWED_MEDIA_TYPES[extension]:
            raise BlobUploadValidationError("Blob content type does not match its extension.")
        if expected_media_type is not None and metadata.content_type != expected_media_type:
            raise BlobUploadValidationError("Blob content type does not match the upload plan.")
        if metadata.size <= 0:
            raise BlobUploadValidationError("The uploaded file is empty.")
        if metadata.size > self.max_bytes:
            raise UploadTooLargeError(
                f"File exceeds the {self.max_bytes // MEBIBYTE} MB limit."
            )
        if expected_size is not None and metadata.size != expected_size:
            raise BlobUploadValidationError("Blob size changed before analysis.")

        self.temp_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="skripsicheck-", dir=self.temp_root
            ) as temporary_directory:
                target = Path(temporary_directory) / f"document{extension}"
                try:
                    self._get_client().download_file(
                        metadata.pathname,
                        target,
                        access="private",
                        overwrite=False,
                        create_parents=False,
                    )
                except Exception as exc:
                    raise BlobStorageError("Vercel Blob download failed.") from exc

                if not target.is_file():
                    raise BlobStorageError("Vercel Blob download did not create a file.")
                if target.stat().st_size != metadata.size:
                    raise BlobUploadValidationError(
                        "Downloaded Blob size does not match its metadata."
                    )
                if validate_document:
                    try:
                        extract_text(target, max_size_mb=self.max_bytes // MEBIBYTE)
                    except DocumentExtractionError as exc:
                        raise BlobUploadValidationError(
                            f"The uploaded document is invalid: {exc}"
                        ) from exc
                yield target
        except OSError as exc:
            raise BlobStorageError("Could not use temporary storage for analysis.") from exc

    def delete(self, pathname: str) -> bool:
        """Delete an object within this adapter's namespace (idempotent in Blob)."""

        self._validate_pathname(pathname)
        try:
            self._get_client().delete(pathname)
        except BlobStorageError:
            raise
        except Exception as exc:
            raise BlobStorageError("Vercel Blob delete failed.") from exc
        return True

    def close(self) -> None:
        """Close a lazily-created SDK transport."""

        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> VercelBlobDocumentStorage:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _get_client(self) -> BlobClientProtocol:
        if self._client is not None:
            return self._client
        token = os.getenv(self.token_env)
        if not token:
            raise BlobStorageConfigurationError(
                f"Private Vercel Blob requires the {self.token_env} environment variable."
            )
        try:
            self._client = self._client_factory(token)
        except BlobStorageConfigurationError:
            raise
        except Exception as exc:
            raise BlobStorageConfigurationError(
                "Could not initialize the Vercel Blob client."
            ) from exc
        return self._client

    def _head(self, pathname: str) -> _BlobMetadata:
        try:
            result = self._get_client().head(pathname)
            metadata = _BlobMetadata(
                pathname=str(result.pathname),
                url=str(result.url),
                content_type=str(result.content_type).split(";", maxsplit=1)[0].strip().lower(),
                size=int(result.size),
                etag=str(result.etag) if getattr(result, "etag", None) else None,
            )
        except BlobStorageError:
            raise
        except Exception as exc:
            raise BlobStorageError("Could not read Vercel Blob metadata.") from exc
        self._validate_private_url(metadata.url)
        self._validate_pathname(metadata.pathname)
        return metadata

    def _validate_completed_upload(
        self, plan: DirectUploadPlan, metadata: _BlobMetadata
    ) -> None:
        if metadata.pathname != plan.pathname:
            raise BlobUploadValidationError("Blob pathname does not match the upload plan.")
        if metadata.content_type != plan.media_type:
            raise BlobUploadValidationError("Blob content type does not match the upload plan.")
        if metadata.size != plan.declared_size_bytes:
            raise BlobUploadValidationError("Blob size does not match the upload plan.")
        if metadata.size > plan.maximum_size_bytes or metadata.size > self.max_bytes:
            raise UploadTooLargeError(
                f"File exceeds the {self.max_bytes // MEBIBYTE} MB limit."
            )

    def _validate_pathname(self, pathname: str) -> None:
        candidate = PurePosixPath(pathname)
        expected_prefix = PurePosixPath(self.prefix)
        if (
            not pathname
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.parts[: len(expected_prefix.parts)] != expected_prefix.parts
            or len(candidate.parts) <= len(expected_prefix.parts)
        ):
            raise BlobUploadValidationError(
                "Refusing to access a Blob outside document storage."
            )

    @staticmethod
    def _validate_private_url(url: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not host.endswith(".private.blob.vercel-storage.com")
            or not unquote(parsed.path).lstrip("/")
        ):
            raise BlobUploadValidationError("Blob URL is not a private Vercel Blob URL.")

    def _delete_quietly(self, pathname: str) -> None:
        try:
            self._get_client().delete(pathname)
        except Exception:
            # Preserve the validation error. Production callers should monitor and
            # periodically clean abandoned objects in the documents/ namespace.
            return
