from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.document_storage import UploadStorageError, UploadTooLargeError
from app.services.vercel_blob_storage import (
    BlobStorageError,
    BlobStorageConfigurationError,
    BlobUploadValidationError,
    VercelBlobDocumentStorage,
)


@dataclass
class FakeMetadata:
    pathname: str
    url: str
    content_type: str
    size: int
    etag: str = '"etag-1"'


class FakeBlobClient:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[FakeMetadata, bytes]] = {}
        self.deleted: list[str] = []
        self.download_access: str | None = None
        self.closed = False

    def add(self, pathname: str, content_type: str, data: bytes) -> None:
        self.objects[pathname] = (
            FakeMetadata(
                pathname=pathname,
                url=(
                    "https://test-store.private.blob.vercel-storage.com/"
                    f"{pathname}"
                ),
                content_type=content_type,
                size=len(data),
            ),
            data,
        )

    def head(self, url_or_path: str, *, token: str | None = None) -> FakeMetadata:
        del token
        return self.objects[url_or_path][0]

    def download_file(
        self,
        url_or_path: str,
        local_path: str | Path,
        *,
        access: str = "public",
        timeout: float | None = None,
        overwrite: bool = True,
        create_parents: bool = True,
        token: str | None = None,
    ) -> str:
        del timeout, overwrite, create_parents, token
        self.download_access = access
        target = Path(local_path)
        target.write_bytes(self.objects[url_or_path][1])
        return str(target)

    def delete(self, url_or_path: str, *, token: str | None = None) -> None:
        del token
        self.deleted.append(url_or_path)
        self.objects.pop(url_or_path, None)

    def close(self) -> None:
        self.closed = True


def make_storage(tmp_path: Path, client: FakeBlobClient) -> VercelBlobDocumentStorage:
    return VercelBlobDocumentStorage(
        max_upload_mb=1,
        temp_root=tmp_path / "tmp",
        client=client,
    )


def test_prepare_direct_upload_produces_safe_private_plan(tmp_path: Path) -> None:
    storage = make_storage(tmp_path, FakeBlobClient())

    plan = storage.prepare_direct_upload(
        "../../skripsi akhir.txt", "text/plain; charset=utf-8", 12
    )

    assert plan.original_filename == "skripsi akhir.txt"
    assert plan.pathname == f"documents/{plan.document_id}.txt"
    assert plan.media_type == "text/plain"
    assert plan.maximum_size_bytes == 1024 * 1024
    assert plan.multipart is True


@pytest.mark.parametrize(
    ("filename", "media_type", "size", "error"),
    [
        ("payload.exe", "application/octet-stream", 10, UploadStorageError),
        ("skripsi.pdf", "text/plain", 10, UploadStorageError),
        ("skripsi.txt", "text/plain", 0, UploadStorageError),
        ("skripsi.txt", "text/plain", 1024 * 1024 + 1, UploadTooLargeError),
    ],
)
def test_prepare_direct_upload_rejects_invalid_browser_metadata(
    tmp_path: Path,
    filename: str,
    media_type: str,
    size: int,
    error: type[Exception],
) -> None:
    storage = make_storage(tmp_path, FakeBlobClient())
    with pytest.raises(error):
        storage.prepare_direct_upload(filename, media_type, size)


def test_finalize_validates_downloaded_document_and_returns_record(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    data = b"Dokumen penelitian yang valid."
    plan = storage.prepare_direct_upload("skripsi.txt", "text/plain", len(data))
    client.add(plan.pathname, plan.media_type, data)

    stored = storage.finalize_direct_upload(plan)

    assert stored.document_id == plan.document_id
    assert stored.stored_path == plan.pathname
    assert stored.size_bytes == len(data)
    assert stored.blob_url.startswith("https://test-store.private.")
    assert client.download_access == "private"
    assert not list((tmp_path / "tmp").iterdir())


def test_finalize_browser_upload_derives_trusted_metadata(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    session_id = "10000000-0000-4000-8000-000000000001"
    document_id = "20000000-0000-4000-8000-000000000002"
    pathname = f"documents/{session_id}/{document_id}.txt"
    data = b"Isi dokumen browser yang valid."
    client.add(pathname, "text/plain", data)

    stored = storage.finalize_browser_upload(
        pathname,
        "../../skripsi akhir.txt",
        session_id=session_id,
    )

    assert stored.document_id == document_id
    assert stored.original_filename == "skripsi akhir.txt"
    assert stored.stored_path == pathname
    assert stored.size_bytes == len(data)


def test_finalize_browser_upload_rejects_another_session(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    pathname = (
        "documents/10000000-0000-4000-8000-000000000001/"
        "20000000-0000-4000-8000-000000000002.txt"
    )

    with pytest.raises(BlobUploadValidationError, match="browser session"):
        storage.finalize_browser_upload(
            pathname,
            "skripsi.txt",
            session_id="30000000-0000-4000-8000-000000000003",
        )

    assert client.deleted == []


def test_finalize_browser_upload_requires_uuid4_document_identifier(
    tmp_path: Path,
) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    session_id = "10000000-0000-4000-8000-000000000001"
    # Canonical but version-1, so it must not bypass the Node token-route
    # contract if this Python endpoint is called independently.
    document_id = "550e8400-e29b-11d4-a716-446655440000"
    pathname = f"documents/{session_id}/{document_id}.txt"

    with pytest.raises(BlobUploadValidationError, match="UUIDv4"):
        storage.finalize_browser_upload(
            pathname,
            "skripsi.txt",
            session_id=session_id,
        )

    assert client.deleted == []


def test_finalize_deletes_blob_when_metadata_does_not_match(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    plan = storage.prepare_direct_upload("skripsi.txt", "text/plain", 5)
    client.add(plan.pathname, plan.media_type, b"different-size")

    with pytest.raises(BlobUploadValidationError, match="size"):
        storage.finalize_direct_upload(plan)

    assert client.deleted == [plan.pathname]


def test_finalize_keeps_blob_when_download_service_is_temporarily_unavailable(
    tmp_path: Path,
) -> None:
    class FailingDownloadClient(FakeBlobClient):
        def download_file(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            raise RuntimeError("temporary network outage")

    client = FailingDownloadClient()
    storage = make_storage(tmp_path, client)
    session_id = "10000000-0000-4000-8000-000000000001"
    document_id = "20000000-0000-4000-8000-000000000002"
    pathname = f"documents/{session_id}/{document_id}.txt"
    client.add(pathname, "text/plain", b"Isi dokumen yang valid.")

    with pytest.raises(BlobStorageError, match="download"):
        storage.finalize_browser_upload(
            pathname,
            "skripsi.txt",
            session_id=session_id,
        )

    assert client.deleted == []
    assert pathname in client.objects


def test_materialize_for_analysis_cleans_temporary_file(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    pathname = "documents/00000000-0000-0000-0000-000000000001.txt"
    data = b"Teks untuk dianalisis."
    client.add(pathname, "text/plain", data)

    with storage.materialize_for_analysis(pathname) as local_path:
        assert local_path.read_bytes() == data
        assert local_path.is_file()

    assert not local_path.exists()
    assert not list((tmp_path / "tmp").iterdir())


def test_materialize_rejects_path_traversal_and_public_url(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)

    with pytest.raises(BlobUploadValidationError, match="outside"):
        with storage.materialize_for_analysis("documents/../private.txt"):
            pass

    pathname = "documents/00000000-0000-0000-0000-000000000001.txt"
    client.add(pathname, "text/plain", b"private")
    metadata, data = client.objects[pathname]
    metadata.url = "https://test-store.public.blob.vercel-storage.com/file.txt"
    client.objects[pathname] = (metadata, data)
    with pytest.raises(BlobUploadValidationError, match="private"):
        with storage.materialize_for_analysis(pathname):
            pass


def test_delete_is_namespace_limited_and_idempotent(tmp_path: Path) -> None:
    client = FakeBlobClient()
    storage = make_storage(tmp_path, client)
    pathname = "documents/00000000-0000-0000-0000-000000000001.txt"

    assert storage.delete(pathname) is True
    assert client.deleted == [pathname]
    with pytest.raises(BlobUploadValidationError):
        storage.delete("other-users/document.txt")


def test_sdk_client_is_lazy_and_token_comes_only_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLOB_READ_WRITE_TOKEN", raising=False)
    created_with: list[str] = []

    def factory(token: str) -> FakeBlobClient:
        created_with.append(token)
        return FakeBlobClient()

    storage = VercelBlobDocumentStorage(
        max_upload_mb=1,
        temp_root=tmp_path,
        client_factory=factory,
    )
    # Preparing a path never needs or exposes the secret.
    storage.prepare_direct_upload("skripsi.txt", "text/plain", 1)
    assert created_with == []

    with pytest.raises(BlobStorageConfigurationError, match="BLOB_READ_WRITE_TOKEN"):
        storage.delete("documents/00000000-0000-0000-0000-000000000001.txt")

    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "test-token-not-for-production")
    assert storage.delete("documents/00000000-0000-0000-0000-000000000001.txt")
    assert created_with == ["test-token-not-for-production"]


def test_blob_finalize_retry_is_idempotent_and_keeps_registered_object(
    tmp_path: Path,
) -> None:
    access_token = "test-access-token-that-is-long-enough-123"
    session_id = "10000000-0000-4000-8000-000000000001"
    document_id = "20000000-0000-4000-8000-000000000002"
    pathname = f"documents/{session_id}/{document_id}.txt"
    blob_client = FakeBlobClient()
    blob_client.add(pathname, "text/plain", b"Isi dokumen yang valid.")
    storage = make_storage(tmp_path, blob_client)
    application = create_app(
        settings=Settings(access_token=access_token),
        database_url=f"sqlite:///{(tmp_path / 'retry.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
    )
    application.state.container.storage = storage
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-SkripsiCheck-Session-ID": session_id,
    }

    with TestClient(application) as client:
        first = client.post(
            "/api/documents/blob",
            headers=headers,
            json={"pathname": pathname, "filename": "skripsi.txt"},
        )
        retry = client.post(
            "/api/documents/blob",
            headers=headers,
            json={"pathname": pathname, "filename": "skripsi.txt"},
        )

    assert first.status_code == 201, first.text
    assert retry.status_code == 201, retry.text
    assert retry.json() == first.json()
    assert blob_client.deleted == []
    assert pathname in blob_client.objects


def test_blob_finalize_reports_storage_outage_as_service_unavailable(
    tmp_path: Path,
) -> None:
    access_token = "test-access-token-that-is-long-enough-123"
    session_id = "10000000-0000-4000-8000-000000000001"
    document_id = "20000000-0000-4000-8000-000000000002"
    pathname = f"documents/{session_id}/{document_id}.txt"
    storage = make_storage(tmp_path, FakeBlobClient())
    application = create_app(
        settings=Settings(access_token=access_token),
        database_url=f"sqlite:///{(tmp_path / 'outage.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
    )
    application.state.container.storage = storage

    with TestClient(application) as client:
        response = client.post(
            "/api/documents/blob",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-SkripsiCheck-Session-ID": session_id,
            },
            json={"pathname": pathname, "filename": "skripsi.txt"},
        )

    assert response.status_code == 503
    assert "metadata" in response.json()["detail"]
