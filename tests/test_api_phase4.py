from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.indexer import NoSimilarityIndexError
from app.main import create_app
from app.services.similarity_engine import RetrievedCandidate


class FakeRetriever:
    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        del top_k
        return [
            RetrievedCandidate(
                chunk_id="source-chunk-1",
                source_file="jurnal_kalsium.txt",
                source_path="references/jurnal_kalsium.txt",
                text=query,
                semantic_score=1.0,
                word_count=len(query.split()),
            )
        ]


class MissingIndexRetriever:
    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        del query, top_k
        raise NoSimilarityIndexError(
            "No similarity index found.\n\nRun:\n\nskripsicheck index ./references"
        )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    application = create_app(
        database_url=f"sqlite:///{(tmp_path / 'test.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
        retriever_factory=FakeRetriever,
    )
    with TestClient(application) as test_client:
        yield test_client


def upload_txt(client: TestClient, text: str, filename: str = "skripsi.txt") -> dict[str, object]:
    response = client.post(
        "/api/documents",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_and_security_headers(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0", "phase": 5}
    assert response.headers["x-content-type-options"] == "nosniff"
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "script-src-attr 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "https://blob.vercel-storage.com" in policy
    assert "https://*.blob.vercel-storage.com" in policy
    assert "'unsafe-inline'" not in policy

    home = client.get("/")
    assert home.headers["content-security-policy"] == policy

    # FastAPI's built-in documentation currently uses an inline bootstrap
    # script, so applying the application CSP here would render it blank.
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "content-security-policy" not in docs.headers


def test_upload_sanitizes_filename_and_persists_metadata(client: TestClient) -> None:
    uploaded = upload_txt(client, "Isi dokumen penelitian.", "../../skripsi akhir.txt")
    assert uploaded["filename"] == "skripsi akhir.txt"
    assert uploaded["extension"] == ".txt"
    assert uploaded["size_bytes"] > 0

    response = client.get(f"/api/documents/{uploaded['id']}")
    assert response.status_code == 200
    assert response.json()["filename"] == "skripsi akhir.txt"
    assert response.headers["cache-control"] == "no-store"


def test_upload_rejects_invalid_extension_mime_and_spoofed_pdf(client: TestClient) -> None:
    unsupported = client.post(
        "/api/documents",
        files={"file": ("malware.exe", b"not executable", "application/octet-stream")},
    )
    assert unsupported.status_code == 422

    bad_mime = client.post(
        "/api/documents",
        files={"file": ("document.txt", b"text", "application/pdf")},
    )
    assert bad_mime.status_code == 422

    spoofed = client.post(
        "/api/documents",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )
    assert spoofed.status_code == 422
    assert "not PDF" in spoofed.json()["detail"]


def test_upload_limit_is_enforced_while_streaming(tmp_path: Path) -> None:
    settings = Settings(max_upload_mb=1)
    application = create_app(
        settings=settings,
        database_url=f"sqlite:///{(tmp_path / 'limit.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
        retriever_factory=FakeRetriever,
    )
    with TestClient(application) as test_client:
        response = test_client.post(
            "/api/documents",
            files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )
    assert response.status_code == 413
    assert not list((tmp_path / "uploads").iterdir())


@pytest.mark.parametrize(
    ("settings", "text", "expected_detail"),
    [
        (
            Settings(max_analysis_characters=30),
            "Dokumen penelitian ini menghasilkan teks yang melampaui batas aman.",
            "characters",
        ),
        (
            Settings(max_analysis_paragraphs=1),
            "Paragraf penelitian pertama.\n\nParagraf penelitian kedua.",
            "paragraphs",
        ),
    ],
)
def test_analysis_limits_fail_fast_before_semantic_retrieval(
    tmp_path: Path,
    settings: Settings,
    text: str,
    expected_detail: str,
) -> None:
    retriever_calls = 0

    class CountingRetriever(FakeRetriever):
        def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
            nonlocal retriever_calls
            retriever_calls += 1
            return super().search(query, top_k)

    application = create_app(
        settings=settings,
        database_url=f"sqlite:///{(tmp_path / 'limits.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
        retriever_factory=CountingRetriever,
    )
    with TestClient(application) as test_client:
        uploaded = upload_txt(test_client, text)
        response = test_client.post(
            "/api/analyses", json={"document_id": uploaded["id"]}
        )

    assert response.status_code == 413
    assert expected_detail in response.json()["detail"]
    assert retriever_calls == 0


def test_analysis_and_persisted_report(client: TestClient) -> None:
    text = (
        "Kalsium membantu memperkuat kualitas cangkang telur puyuh.\n\n"
        "Kalsium membantu memperkuat kualitas cangkang telur puyuh."
    )
    uploaded = upload_txt(client, text)

    analysis = client.post("/api/analyses", json={"document_id": uploaded["id"]})

    assert analysis.status_code == 201, analysis.text
    created = analysis.json()
    # The same source chunk is counted once across two identical paragraphs.
    assert created["overall_similarity"] == pytest.approx(0.5)
    assert created["total_paragraphs"] == 2
    assert created["matched_paragraphs"] == 2

    report = client.get(created["report_url"])
    assert report.status_code == 200
    payload = report.json()
    assert payload["analysis_id"] == created["id"]
    assert len(payload["paragraphs"]) == 2
    assert payload["paragraphs"][0]["matches"][0]["semantic_similarity"] == 1.0
    assert "not an automated plagiarism" in payload["disclaimer"]


def test_analysis_reports_missing_index_as_conflict(tmp_path: Path) -> None:
    application = create_app(
        database_url=f"sqlite:///{(tmp_path / 'missing.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
        retriever_factory=MissingIndexRetriever,
    )
    with TestClient(application) as test_client:
        uploaded = upload_txt(test_client, "Dokumen dengan isi yang cukup.")
        response = test_client.post(
            "/api/analyses", json={"document_id": uploaded["id"]}
        )
    assert response.status_code == 409
    assert "skripsicheck index" in response.json()["detail"]


def test_delete_document_removes_metadata_and_file(client: TestClient) -> None:
    uploaded = upload_txt(client, "Dokumen sementara.")
    response = client.delete(f"/api/documents/{uploaded['id']}")
    assert response.status_code == 204
    assert client.get(f"/api/documents/{uploaded['id']}").status_code == 404
