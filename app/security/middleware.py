"""ASGI middleware alternative for applying the shared pilot access gate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.security.access import (
    AUTHORIZATION_SCHEME,
    SharedAccessGuard,
    access_token_from_environment,
)


def _normalize_path_prefix(prefix: str) -> str:
    normalized = "/" + prefix.strip("/")
    return normalized if normalized != "/" else "/"


def _path_matches(path: str, prefix: str) -> bool:
    return prefix == "/" or path == prefix or path.startswith(f"{prefix}/")


class SharedAccessMiddleware:
    """Protect selected HTTP path prefixes with the optional shared token.

    By default the token is read once from ``SKRIPSICHECK_ACCESS_TOKEN`` when
    the middleware stack is constructed. If it is absent, the middleware is a
    no-op. Pass ``load_from_environment=False`` to explicitly disable lookup.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        guard: SharedAccessGuard | None = None,
        expected_token: str | None = None,
        load_from_environment: bool = True,
        environment: Mapping[str, str] | None = None,
        protected_path_prefixes: Iterable[str] = ("/api",),
        excluded_paths: Iterable[str] = ("/health",),
    ) -> None:
        self.app = app
        if guard is not None:
            self._guard = guard
        else:
            configured_token = expected_token
            if configured_token is None and load_from_environment:
                configured_token = access_token_from_environment(environment)
            self._guard = SharedAccessGuard(configured_token)
        self._protected_prefixes = tuple(
            _normalize_path_prefix(prefix) for prefix in protected_path_prefixes
        )
        self._excluded_paths = frozenset(
            _normalize_path_prefix(path) for path in excluded_paths
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._guard.enabled:
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", "/"))
        protected = any(
            _path_matches(path, prefix) for prefix in self._protected_prefixes
        )
        if path in self._excluded_paths or not protected:
            await self.app(scope, receive, send)
            return

        authorization = Headers(scope=scope).get("Authorization")
        if self._guard.accepts_authorization(authorization):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"detail": "Access denied."},
            status_code=401,
            headers={"WWW-Authenticate": AUTHORIZATION_SCHEME},
        )
        await response(scope, receive, send)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(enabled={self._guard.enabled!r}, "
            f"protected_path_prefixes={self._protected_prefixes!r})"
        )


__all__ = ["SharedAccessMiddleware"]
