from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.database.repository import Repository
from app.database.session import Database
from app.main import create_app
from app.security import BROWSER_SESSION_HEADER
from app.services.similarity_engine import RetrievedCandidate


ACCESS_TOKEN = "a-long-private-pilot-access-token"


class FakeRetriever:
    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        del top_k
        return [
            RetrievedCandidate(
                chunk_id="source-chunk-1",
                source_file="reference.txt",
                source_path="references/reference.txt",
                text=query,
                semantic_score=1.0,
                word_count=len(query.split()),
            )
        ]


def _pilot_headers(session_id: str, token: str = ACCESS_TOKEN) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        BROWSER_SESSION_HEADER: session_id,
    }


def _pilot_app(tmp_path: Path, *, max_upload_mb: int = 25):
    settings = Settings(access_token=ACCESS_TOKEN, max_upload_mb=max_upload_mb)
    return create_app(
        settings=settings,
        database_url=f"sqlite:///{(tmp_path / 'pilot.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
        retriever_factory=FakeRetriever,
    )


def test_settings_repr_never_contains_access_token() -> None:
    settings = Settings(access_token=ACCESS_TOKEN)
    assert ACCESS_TOKEN not in repr(settings)


def test_runtime_is_public_and_reports_only_safe_capabilities(tmp_path: Path) -> None:
    application = _pilot_app(tmp_path, max_upload_mb=7)
    assert ACCESS_TOKEN not in repr(application.user_middleware)
    with TestClient(application) as client:
        response = client.get("/api/runtime")

    assert response.status_code == 200
    assert response.json() == {
        "access_required": True,
        "direct_upload": False,
        "max_upload_mb": 7,
    }
    assert ACCESS_TOKEN not in response.text


def test_protected_pilot_disables_interactive_api_schema(tmp_path: Path) -> None:
    application = _pilot_app(tmp_path)
    with TestClient(application) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_protected_api_requires_token_then_uuid4_session(tmp_path: Path) -> None:
    application = _pilot_app(tmp_path)
    file_payload = {"file": ("paper.txt", b"Academic text.", "text/plain")}
    with TestClient(application) as client:
        missing_token = client.post("/api/documents", files=file_payload)
        wrong_token = client.post(
            "/api/documents",
            files=file_payload,
            headers=_pilot_headers(str(uuid4()), token="wrong"),
        )
        missing_session = client.post(
            "/api/documents",
            files=file_payload,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
        )
        invalid_session = client.post(
            "/api/documents",
            files=file_payload,
            headers=_pilot_headers("not-a-uuid"),
        )

    assert missing_token.status_code == 401
    assert wrong_token.status_code == 401
    assert missing_session.status_code == 400
    assert invalid_session.status_code == 400
    assert ACCESS_TOKEN not in missing_token.text


def test_cross_session_resources_always_look_missing(tmp_path: Path) -> None:
    application = _pilot_app(tmp_path)
    owner_a = str(uuid4())
    owner_b = str(uuid4())
    headers_a = _pilot_headers(owner_a)
    headers_b = _pilot_headers(owner_b)

    with TestClient(application) as client:
        upload = client.post(
            "/api/documents",
            files={
                "file": (
                    "paper.txt",
                    b"Calcium supports strong eggshell quality.",
                    "text/plain",
                )
            },
            headers=headers_a,
        )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["id"]

        assert client.get(
            f"/api/documents/{document_id}", headers=headers_b
        ).status_code == 404
        assert client.delete(
            f"/api/documents/{document_id}", headers=headers_b
        ).status_code == 404
        assert client.post(
            "/api/analyses",
            json={"document_id": document_id},
            headers=headers_b,
        ).status_code == 404

        analysis = client.post(
            "/api/analyses",
            json={"document_id": document_id},
            headers=headers_a,
        )
        assert analysis.status_code == 201, analysis.text
        report_url = analysis.json()["report_url"]
        assert client.get(report_url, headers=headers_b).status_code == 404
        assert client.get(report_url, headers=headers_a).status_code == 200

        # A failed cross-session delete must not remove the owner's document.
        assert client.get(
            f"/api/documents/{document_id}", headers=headers_a
        ).status_code == 200
        assert client.delete(
            f"/api/documents/{document_id}", headers=headers_a
        ).status_code == 204


def test_existing_sqlite_documents_migrate_to_local_owner(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    database = Database(f"sqlite:///{database_path.as_posix()}")
    try:
        with database.engine.begin() as connection:
            connection.exec_driver_sql(
                """
                CREATE TABLE documents (
                    id VARCHAR(36) NOT NULL PRIMARY KEY,
                    original_filename VARCHAR(255) NOT NULL,
                    stored_path TEXT NOT NULL UNIQUE,
                    media_type VARCHAR(127) NOT NULL,
                    extension VARCHAR(10) NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at DATETIME NOT NULL
                )
                """
            )
            connection.exec_driver_sql(
                """
                INSERT INTO documents (
                    id, original_filename, stored_path, media_type,
                    extension, size_bytes, created_at
                ) VALUES (
                    'legacy-document', 'legacy.txt', 'uploads/legacy.txt',
                    'text/plain', '.txt', 4, '2026-08-18 00:00:00'
                )
                """
            )

        database.create_schema()
        repository = Repository(database)
        migrated = repository.get_document("legacy-document")
        assert migrated is not None
        assert migrated.owner_session_id == "local"
        assert repository.get_document("legacy-document", str(uuid4())) is None

        with database.session() as session:
            columns = {
                row[1]
                for row in session.execute(
                    text("PRAGMA table_info(documents)")
                ).all()
            }
            indexes = {
                row[1]
                for row in session.execute(
                    text("PRAGMA index_list(documents)")
                ).all()
            }
        assert "owner_session_id" in columns
        assert "ix_documents_owner_session_id" in indexes

        # The migration is deliberately idempotent.
        database.create_schema()
    finally:
        database.close()
