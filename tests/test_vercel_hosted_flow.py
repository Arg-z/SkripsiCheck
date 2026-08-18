"""Simulated end-to-end coverage for the private Vercel deployment flow.

The test deliberately uses only in-memory/fake hosted services and a temporary
SQLite database.  It must never require Vercel Blob, Neon, FAISS artifacts, or
the Sentence Transformers model to be available over the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.security import BROWSER_SESSION_HEADER
from app.services.similarity_engine import RetrievedCandidate
from app.services.vercel_blob_storage import VercelBlobDocumentStorage


ACCESS_TOKEN = "hosted-pilot-access-token-at-least-32-characters"


@dataclass(slots=True)
class FakeBlobMetadata:
    pathname: str
    url: str
    content_type: str
    size: int
    etag: str = '"hosted-e2e-etag"'


class FakePrivateBlobClient:
    """Minimal private Blob transport used by the real storage adapter."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[FakeBlobMetadata, bytes]] = {}
        self.downloads: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.closed = False

    def add(self, pathname: str, content_type: str, payload: bytes) -> None:
        self.objects[pathname] = (
            FakeBlobMetadata(
                pathname=pathname,
                url=(
                    "https://skripsicheck-test.private.blob.vercel-storage.com/"
                    f"{pathname}"
                ),
                content_type=content_type,
                size=len(payload),
            ),
            payload,
        )

    def head(self, url_or_path: str, *, token: str | None = None) -> FakeBlobMetadata:
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
        self.downloads.append((url_or_path, access))
        target = Path(local_path)
        target.write_bytes(self.objects[url_or_path][1])
        return str(target)

    def delete(self, url_or_path: str, *, token: str | None = None) -> None:
        del token
        self.deleted.append(url_or_path)
        self.objects.pop(url_or_path, None)

    def close(self) -> None:
        self.closed = True


class FakeSourceIndex:
    """Deterministic semantic index standing in for cached FAISS artifacts."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        self.queries.append((query, top_k))
        return [
            RetrievedCandidate(
                chunk_id="hosted-reference-chunk",
                source_file="jurnal_kalsium.txt",
                source_path="private-index/jurnal_kalsium.txt",
                text=query,
                semantic_score=1.0,
                word_count=len(query.split()),
            )
        ][:top_k]


class FakeIndexCache:
    """Warm-function cache boundary around the fake source index."""

    def __init__(self, index: FakeSourceIndex) -> None:
        self.index = index
        self.acquisitions = 0
        self.closed = False

    def acquire(self) -> FakeSourceIndex:
        self.acquisitions += 1
        return self.index

    def close(self) -> None:
        self.closed = True


class FakeCachedRetriever:
    """Candidate retriever that obtains its index through the fake cache."""

    def __init__(self, cache: FakeIndexCache) -> None:
        self.cache = cache

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        return self.cache.acquire().search(query, top_k)


def _headers(session_id: str, *, token: str = ACCESS_TOKEN) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        BROWSER_SESSION_HEADER: session_id,
    }


def test_private_hosted_flow_is_authenticated_scoped_and_local_only(
    tmp_path: Path,
) -> None:
    """Exercise runtime -> Blob finalize -> analysis -> report -> delete."""

    owner_session = str(uuid4())
    other_session = str(uuid4())
    document_id = str(uuid4())
    pathname = f"documents/{owner_session}/{document_id}.txt"
    document_bytes = (
        "Kalsium membantu pembentukan cangkang telur puyuh yang kuat.\n\n"
        "Kualitas nutrisi pakan memengaruhi mutu telur yang dihasilkan."
    ).encode("utf-8")

    blob_client = FakePrivateBlobClient()
    blob_client.add(pathname, "text/plain", document_bytes)
    blob_storage = VercelBlobDocumentStorage(
        max_upload_mb=2,
        temp_root=tmp_path / "function-tmp",
        client=blob_client,
    )
    source_index = FakeSourceIndex()
    index_cache = FakeIndexCache(source_index)
    retriever = FakeCachedRetriever(index_cache)
    settings = Settings(
        access_token=ACCESS_TOKEN,
        storage_backend="vercel_blob",
        index_backend="vercel_blob",
        max_upload_mb=2,
    )
    settings.validate()
    application = create_app(
        settings=settings,
        database_url=f"sqlite:///{(tmp_path / 'hosted.sqlite3').as_posix()}",
        retriever_factory=lambda: retriever,
    )

    # Keep the production Blob adapter and analysis orchestration while swapping
    # only their transport/cache boundaries. No credential or network is used.
    container = application.state.container
    container.storage.close()
    container.storage = blob_storage
    container.analysis_service.storage = blob_storage
    container.index_cache = index_cache  # type: ignore[assignment]

    with TestClient(application) as client:
        runtime = client.get("/api/runtime")
        assert runtime.status_code == 200
        assert runtime.json() == {
            "access_required": True,
            "direct_upload": True,
            "max_upload_mb": 2,
        }
        assert ACCESS_TOKEN not in runtime.text

        unauthorized = client.post(
            "/api/documents/blob",
            json={"pathname": pathname, "filename": "skripsi.txt"},
        )
        assert unauthorized.status_code == 401
        assert ACCESS_TOKEN not in unauthorized.text

        missing_session = client.post(
            "/api/documents/blob",
            json={"pathname": pathname, "filename": "skripsi.txt"},
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
        )
        assert missing_session.status_code == 400

        wrong_owner_finalize = client.post(
            "/api/documents/blob",
            json={"pathname": pathname, "filename": "skripsi.txt"},
            headers=_headers(other_session),
        )
        assert wrong_owner_finalize.status_code == 422
        assert pathname in blob_client.objects

        finalized = client.post(
            "/api/documents/blob",
            json={"pathname": pathname, "filename": "skripsi.txt"},
            headers=_headers(owner_session),
        )
        assert finalized.status_code == 201, finalized.text
        assert finalized.json() == {
            "id": document_id,
            "filename": "skripsi.txt",
            "media_type": "text/plain",
            "extension": ".txt",
            "size_bytes": len(document_bytes),
            "created_at": finalized.json()["created_at"],
        }

        # Resource existence is not disclosed across anonymous browser sessions.
        assert client.get(
            f"/api/documents/{document_id}", headers=_headers(other_session)
        ).status_code == 404
        assert client.post(
            "/api/analyses",
            json={"document_id": document_id, "top_k": 3, "min_score": 0.4},
            headers=_headers(other_session),
        ).status_code == 404
        assert client.delete(
            f"/api/documents/{document_id}", headers=_headers(other_session)
        ).status_code == 404
        assert pathname in blob_client.objects

        analysis = client.post(
            "/api/analyses",
            json={"document_id": document_id, "top_k": 3, "min_score": 0.4},
            headers=_headers(owner_session),
        )
        assert analysis.status_code == 201, analysis.text
        assert analysis.json()["document_id"] == document_id
        assert analysis.json()["total_paragraphs"] == 2
        assert analysis.json()["matched_paragraphs"] == 2
        # Both paragraphs intentionally hit the same source chunk. The overall
        # calculation counts that source contribution once instead of inflating
        # the document score with a duplicate hit.
        assert analysis.json()["overall_similarity"] == 0.5
        report_url = analysis.json()["report_url"]

        assert client.get(report_url, headers=_headers(other_session)).status_code == 404
        report = client.get(report_url, headers=_headers(owner_session))
        assert report.status_code == 200, report.text
        assert report.json()["document"] == {
            "id": document_id,
            "filename": "skripsi.txt",
        }
        assert report.json()["overall_similarity"] == 0.5
        assert [
            paragraph["matches"][0]["source_file"]
            for paragraph in report.json()["paragraphs"]
        ] == ["jurnal_kalsium.txt", "jurnal_kalsium.txt"]
        assert all(
            paragraph["matches"][0]["source_path"] is None
            for paragraph in report.json()["paragraphs"]
        )

        assert client.delete(
            f"/api/documents/{document_id}", headers=_headers(owner_session)
        ).status_code == 204
        assert client.get(
            f"/api/documents/{document_id}", headers=_headers(owner_session)
        ).status_code == 404
        assert client.get(report_url, headers=_headers(owner_session)).status_code == 404

    assert source_index.queries == [
        ("Kalsium membantu pembentukan cangkang telur puyuh yang kuat.", 3),
        ("Kualitas nutrisi pakan memengaruhi mutu telur yang dihasilkan.", 3),
    ]
    assert index_cache.acquisitions == 2
    assert index_cache.closed is True
    assert blob_client.downloads == [(pathname, "private"), (pathname, "private")]
    assert blob_client.deleted == [pathname]
    assert blob_client.objects == {}
    assert blob_client.closed is False  # injected clients remain owned by their caller
    assert not list((tmp_path / "function-tmp").iterdir())
