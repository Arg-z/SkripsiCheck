from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.security import (
    ACCESS_TOKEN_ENV,
    BROWSER_SESSION_HEADER,
    SharedAccessGuard,
    SharedAccessMiddleware,
    access_token_from_environment,
    parse_browser_session_id,
    require_browser_session,
    secure_token_matches,
)


def test_optional_access_token_defaults_to_disabled() -> None:
    assert access_token_from_environment({}) is None
    assert access_token_from_environment({ACCESS_TOKEN_ENV: "   "}) is None
    assert SharedAccessGuard.from_environment({}).enabled is False


def test_secure_token_comparison_uses_constant_time_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bytes, bytes]] = []

    def fake_compare(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr("app.security.access.secrets.compare_digest", fake_compare)
    assert secure_token_matches("rahasia", "rahasia") is True
    assert secure_token_matches("salah", "rahasia") is False
    assert calls
    assert all(len(left) == len(right) == 32 for left, right in calls)


def test_guard_allows_local_mode_and_valid_bearer_but_rejects_invalid() -> None:
    local_app = FastAPI()
    local_guard = SharedAccessGuard(None)

    @local_app.get("/private", dependencies=[Depends(local_guard)])
    def local_private() -> dict[str, bool]:
        return {"ok": True}

    assert TestClient(local_app).get("/private").status_code == 200

    protected_app = FastAPI()
    secret = "pilot-secret-value"
    guard = SharedAccessGuard(secret)

    @protected_app.get("/private", dependencies=[Depends(guard)])
    def protected_private() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(protected_app)
    denied = client.get("/private", headers={"Authorization": "Bearer wrong"})
    assert denied.status_code == 401
    assert denied.json() == {"detail": "Access denied."}
    assert denied.headers["www-authenticate"] == "Bearer"
    assert secret not in denied.text
    assert secret not in repr(guard)

    allowed = client.get(
        "/private",
        headers={"Authorization": f"Bearer {secret}"},
    )
    assert allowed.status_code == 200


def test_browser_session_parser_accepts_only_canonical_uuid4() -> None:
    session_id = uuid4()
    assert parse_browser_session_id(str(session_id)) == session_id
    assert parse_browser_session_id(str(session_id).upper()) == session_id

    with pytest.raises(ValueError):
        parse_browser_session_id(None)
    with pytest.raises(ValueError):
        parse_browser_session_id("not-a-session")
    with pytest.raises(ValueError):
        parse_browser_session_id("550e8400-e29b-11d4-a716-446655440000")
    with pytest.raises(ValueError):
        parse_browser_session_id(f"{{{session_id}}}")


def test_browser_session_dependency_returns_uuid_without_reflecting_bad_input() -> None:
    app = FastAPI()

    @app.get("/owned")
    def owned(
        session_id: Annotated[UUID, Depends(require_browser_session)],
    ) -> dict[str, str]:
        return {"session_id": str(session_id)}

    client = TestClient(app)
    session_id = uuid4()
    accepted = client.get(
        "/owned",
        headers={BROWSER_SESSION_HEADER: str(session_id)},
    )
    assert accepted.status_code == 200
    assert accepted.json() == {"session_id": str(session_id)}

    untrusted = "definitely-not-a-uuid"
    rejected = client.get("/owned", headers={BROWSER_SESSION_HEADER: untrusted})
    assert rejected.status_code == 400
    assert untrusted not in rejected.text


def test_shared_access_middleware_protects_only_configured_prefixes() -> None:
    app = FastAPI()

    @app.get("/")
    def home() -> dict[str, bool]:
        return {"public": True}

    @app.get("/api/items")
    def items() -> dict[str, bool]:
        return {"private": True}

    app.add_middleware(
        SharedAccessMiddleware,
        expected_token="middleware-secret",
        protected_path_prefixes=("/api",),
    )
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/items").status_code == 401
    accepted = client.get(
        "/api/items",
        headers={"Authorization": "Bearer middleware-secret"},
    )
    assert accepted.status_code == 200


def test_shared_access_middleware_is_noop_without_environment_token() -> None:
    app = FastAPI()

    @app.get("/api/items")
    def items() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(
        SharedAccessMiddleware,
        environment={},
        protected_path_prefixes=("/api",),
    )
    assert TestClient(app).get("/api/items").status_code == 200
