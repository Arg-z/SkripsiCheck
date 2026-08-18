"""Reusable security primitives for local and hosted SkripsiCheck deployments.

The package is intentionally independent from the application container and
database so routes can adopt it without coupling authentication to storage.
"""

from app.security.access import (
    ACCESS_TOKEN_ENV,
    SharedAccessGuard,
    access_token_from_environment,
    require_shared_access,
    secure_token_matches,
)
from app.security.middleware import SharedAccessMiddleware
from app.security.session import (
    BROWSER_SESSION_HEADER,
    parse_browser_session_id,
    require_browser_session,
)

__all__ = [
    "ACCESS_TOKEN_ENV",
    "BROWSER_SESSION_HEADER",
    "SharedAccessGuard",
    "SharedAccessMiddleware",
    "access_token_from_environment",
    "parse_browser_session_id",
    "require_browser_session",
    "require_shared_access",
    "secure_token_matches",
]
