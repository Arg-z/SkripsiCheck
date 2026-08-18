"""FastAPI application entrypoint for SkripsiCheck."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import analysis, documents, reports
from app.config import SETTINGS, Settings
from app.models.schemas import HealthResponse
from app.services.container import AppContainer, build_container
from app.services.similarity_engine import CandidateRetriever


APP_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR.parent / "static"


def create_app(
    *,
    settings: Settings = SETTINGS,
    database_url: str | None = None,
    upload_dir: str | Path | None = None,
    retriever_factory: Callable[[], CandidateRetriever] | None = None,
) -> FastAPI:
    container = build_container(
        settings=settings,
        database_url=database_url,
        upload_dir=upload_dir,
        retriever_factory=retriever_factory,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        container.database.close()

    application = FastAPI(
        title="SkripsiCheck",
        version=__version__,
        description=(
            "Local-first text similarity analysis API. Results require manual academic review."
        ),
        lifespan=lifespan,
    )
    application.state.container = container
    application.include_router(documents.router)
    application.include_router(analysis.router)
    application.include_router(reports.router)
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
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, phase=5)

    return application


app = create_app()
