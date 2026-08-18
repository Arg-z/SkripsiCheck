"""PHASE 5 web-shell tests.

These tests deliberately exercise the frontend only through its public HTTP
contract.  They do not need a semantic model or an existing FAISS index.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def web_client(tmp_path: Path) -> TestClient:
    application = create_app(
        database_url=f"sqlite:///{(tmp_path / 'frontend.sqlite3').as_posix()}",
        upload_dir=tmp_path / "uploads",
    )
    with TestClient(application) as client:
        yield client


def test_homepage_exposes_primary_student_workflow(web_client: TestClient) -> None:
    response = web_client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "SkripsiCheck" in response.text
    for control_id in (
        "analyze-button",
        "add-sources-button",
        "rebuild-button",
        "export-json-button",
        "export-html-button",
    ):
        assert f'id="{control_id}"' in response.text
    assert 'href="/static/styles.css"' in response.text
    assert 'src="/static/blob-client.js"' in response.text
    assert 'src="/static/app.js"' in response.text
    assert 'id="access-panel"' in response.text
    assert "https://*.blob.vercel-storage.com" in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/styles.css", "text/css"),
        ("/static/app.js", "javascript"),
        ("/static/blob-client.js", "javascript"),
    ],
)
def test_primary_static_assets_are_served(
    web_client: TestClient,
    path: str,
    content_type: str,
) -> None:
    response = web_client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.content.strip()
    assert response.headers["x-content-type-options"] == "nosniff"


def test_web_responses_keep_security_headers_and_do_not_reflect_query_input(
    web_client: TestClient,
) -> None:
    marker = "frontend-xss-marker"
    response = web_client.get(
        "/",
        params={"filename": f'<img src=x onerror=alert("{marker}")>'},
    )

    assert response.status_code == 200
    assert marker not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_frontend_renders_untrusted_api_data_without_inner_html(
    web_client: TestClient,
) -> None:
    """Guard the DOM rendering contract against introducing an easy XSS sink."""

    response = web_client.get("/static/app.js")
    javascript = response.text

    assert response.status_code == 200
    for unsafe_sink in (".innerHTML", ".outerHTML", "insertAdjacentHTML", "document.write"):
        assert unsafe_sink not in javascript
    assert "textContent" in javascript
    assert "createElement" in javascript


def test_frontend_supports_runtime_access_and_direct_blob_upload(
    web_client: TestClient,
) -> None:
    javascript = web_client.get("/static/app.js").text

    assert 'requestJson("/api/runtime"' in javascript
    assert "window.sessionStorage" in javascript
    assert "window.localStorage" not in javascript
    assert 'headers.set("Authorization", `Bearer ${state.accessToken}`)' in javascript
    assert 'headers.set("X-SkripsiCheck-Session-ID", ensureSessionId())' in javascript
    assert "blobClient.uploadPrivateDocument" in javascript
    assert 'requestJson("/api/documents/blob"' in javascript
    assert 'requestJson("/api/documents"' in javascript
