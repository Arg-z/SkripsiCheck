"""Small FastAPI dependency helpers."""

from fastapi import Request

from app.security.session import require_browser_session
from app.services.container import AppContainer


LOCAL_OWNER_SESSION_ID = "local"


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_owner_session_id(request: Request) -> str:
    """Return the scoped owner, requiring UUIDv4 only for protected pilots."""

    if not bool(getattr(request.app.state, "access_required", False)):
        return LOCAL_OWNER_SESSION_ID
    return str(require_browser_session(request))
