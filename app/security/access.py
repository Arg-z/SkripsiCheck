"""Optional shared-token protection for a small private pilot.

When ``SKRIPSICHECK_ACCESS_TOKEN`` is unset or blank, the dependency permits
requests. This keeps the default local development workflow unchanged. A
hosted pilot can set a strong random value and require ``Authorization:
Bearer <token>`` on protected routes.

This is intentionally a small-pilot access gate, not a user authentication
system. It should eventually be replaced by real accounts and sessions.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Mapping

from fastapi import HTTPException, Request, status


ACCESS_TOKEN_ENV = "SKRIPSICHECK_ACCESS_TOKEN"
AUTHORIZATION_SCHEME = "Bearer"
_ACCESS_DENIED_DETAIL = "Access denied."


def _normalize_configured_token(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def access_token_from_environment(
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Read the optional pilot token without logging or otherwise exposing it."""

    source = os.environ if environment is None else environment
    return _normalize_configured_token(source.get(ACCESS_TOKEN_ENV))


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def secure_token_matches(candidate: str | None, expected: str | None) -> bool:
    """Compare access tokens with a fixed-size, timing-safe comparison."""

    normalized_expected = _normalize_configured_token(expected)
    if candidate is None or normalized_expected is None:
        return False
    return secrets.compare_digest(
        _token_digest(candidate),
        _token_digest(normalized_expected),
    )


def bearer_token_from_authorization(value: str | None) -> str | None:
    """Return a strict Bearer credential, or ``None`` for malformed input."""

    if value is None:
        return None
    parts = value.split()
    if len(parts) != 2 or parts[0].casefold() != AUTHORIZATION_SCHEME.casefold():
        return None
    credential = parts[1]
    return credential if credential else None


def access_denied_exception() -> HTTPException:
    """Build a generic response that never reflects a supplied credential."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=_ACCESS_DENIED_DETAIL,
        headers={"WWW-Authenticate": AUTHORIZATION_SCHEME},
    )


class SharedAccessGuard:
    """FastAPI dependency implementing an optional shared Bearer token gate.

    Only a SHA-256 digest is retained, keeping the configured token out of the
    object's representation and reducing accidental disclosure during logging.
    """

    __slots__ = ("_expected_digest",)

    def __init__(self, expected_token: str | None) -> None:
        normalized = _normalize_configured_token(expected_token)
        self._expected_digest = _token_digest(normalized) if normalized else None

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> SharedAccessGuard:
        return cls(access_token_from_environment(environment))

    @property
    def enabled(self) -> bool:
        return self._expected_digest is not None

    def accepts_authorization(self, authorization: str | None) -> bool:
        """Check one Authorization header without returning secret details."""

        if self._expected_digest is None:
            return True
        candidate = bearer_token_from_authorization(authorization)
        if candidate is None:
            return False
        return secrets.compare_digest(_token_digest(candidate), self._expected_digest)

    def __call__(self, request: Request) -> None:
        if not self.accepts_authorization(request.headers.get("Authorization")):
            raise access_denied_exception()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(enabled={self.enabled!r})"


def require_shared_access(request: Request) -> None:
    """FastAPI dependency using the current environment configuration.

    Routes can use ``Depends(require_shared_access)``. For a stable per-app
    snapshot, create ``SharedAccessGuard.from_environment()`` during app setup
    and use that instance as the dependency instead.
    """

    SharedAccessGuard.from_environment()(request)
