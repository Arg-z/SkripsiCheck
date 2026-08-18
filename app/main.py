"""FastAPI application entrypoint for SkripsiCheck."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import analysis, documents, reports, runtime
from app.config import SETTINGS, Settings
from app.models.schemas import HealthResponse
from app.security.access import SharedAccessGuard
from app.security.middleware import SharedAccessMiddleware
from app.services.container import AppContainer, build_container
from app.services.similarity_engine import CandidateRetriever


APP_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR.parent / "static"
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self'",
        "script-src-attr 'none'",
        "style-src 'self'",
        "style-src-attr 'none'",
        "img-src 'self' data:",
        "font-src 'self'",
        (
            "connect-src 'self' https://blob.vercel-storage.com "
            "https://*.blob.vercel-storage.com"
        ),
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    )
)


def create_app(
    *,
    settings: Settings = SETTINGS,
    database_url: str | None = None,
    upload_dir: str | Path | None = None,
    retriever_factory: Callable[[], CandidateRetriever] | None = None,
) -> FastAPI:
    access_required = bool(settings.access_token and settings.access_token.strip())
    container = build_container(
        settings=settings,
        database_url=database_url,
        upload_dir=upload_dir,
        retriever_factory=retriever_factory,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        container.storage.close()
        if container.index_cache is not None:
            container.index_cache.close()
        container.database.close()

    application = FastAPI(
        title="SkripsiCheck",
        version=__version__,
        description=(
            "Local-first text similarity analysis API. Results require manual academic review."
        ),
        lifespan=lifespan,
        docs_url=None if access_required else "/docs",
        redoc_url=None if access_required else "/redoc",
        openapi_url=None if access_required else "/openapi.json",
    )
    application.state.container = container
    application.state.access_required = access_required
    application.state.direct_upload = settings.storage_backend == "vercel_blob"
    application.state.max_upload_mb = settings.max_upload_mb
    application.include_router(documents.router)
    application.include_router(analysis.router)
    application.include_router(reports.router)
    application.include_router(runtime.router)
    application.add_middleware(
        SharedAccessMiddleware,
        guard=SharedAccessGuard(settings.access_token),
        protected_path_prefixes=("/api",),
        excluded_paths=("/api/runtime",),
    )
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Serve the dependency-free browser interface."""

        return FileResponse(TEMPLATE_DIR / "index.html", media_type="text/html")

    @application.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # FastAPI's stock Swagger/ReDoc pages contain their own inline bootstrap
        # script. Keep those developer pages functional; the student UI and API
        # responses receive the strict no-inline policy below.
        if request.url.path not in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, phase=5)

    return application


app = create_app()
