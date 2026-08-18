"""Database lifecycle for local SQLite and serverless PostgreSQL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.database.entities import Base


def normalize_database_url(url: str) -> str:
    """Return a synchronous SQLAlchemy URL suitable for the installed drivers.

    Neon and other hosted PostgreSQL providers commonly expose ``postgres://``
    or ``postgresql://`` URLs.  SQLAlchemy otherwise assumes the legacy
    psycopg2 driver for those URLs, while SkripsiCheck uses Psycopg 3.
    Explicit driver URLs are left unchanged.
    """

    normalized = url.strip()
    if not normalized:
        raise ValueError("Database URL cannot be empty.")

    scheme, separator, remainder = normalized.partition(":")
    if separator and scheme.lower() in {"postgres", "postgresql"}:
        return f"postgresql+psycopg:{remainder}"
    return normalized


def database_engine_options(url: str) -> dict[str, Any]:
    """Build backend-specific SQLAlchemy options without opening a connection."""

    normalized_url = normalize_database_url(url)
    try:
        backend = make_url(normalized_url).get_backend_name()
    except ArgumentError:
        # Do not echo a malformed connection URL because it may contain a
        # database password or provider token.
        raise ValueError("Database URL is invalid.") from None
    options: dict[str, Any] = {"pool_pre_ping": True}

    if backend == "sqlite":
        options["connect_args"] = {"check_same_thread": False}
    elif backend == "postgresql":
        # Serverless instances can be frozen or multiplied at any time.  A
        # process-local persistent pool can therefore hold stale connections
        # or multiply the number of open database connections.  Neon performs
        # pooling at its pooled endpoint, so each SQLAlchemy checkout can use a
        # short-lived DBAPI connection here.
        options["poolclass"] = NullPool

    return options


class Database:
    """Own the SQLAlchemy engine and short-lived session factory."""

    def __init__(self, url: str) -> None:
        self.url = normalize_database_url(url)
        self._ensure_sqlite_parent(self.url)
        engine_options = database_engine_options(self.url)
        self.engine: Engine = create_engine(
            self.url,
            **engine_options,
        )
        if "connect_args" in engine_options:
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.session_factory = sessionmaker(
            bind=self.engine, class_=Session, expire_on_commit=False
        )

    @staticmethod
    def _ensure_sqlite_parent(url: str) -> None:
        prefix = "sqlite:///"
        if not url.startswith(prefix) or url == "sqlite:///:memory:":
            return
        path = Path(url.removeprefix(prefix))
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        self._migrate_sqlite_document_ownership()

    def _migrate_sqlite_document_ownership(self) -> None:
        """Add anonymous ownership to databases created before the pilot.

        ``create_all`` intentionally does not alter existing tables. This
        narrow, idempotent SQLite migration preserves every existing document
        under the local owner while allowing new installations to use the
        declarative schema directly.
        """

        if self.engine.dialect.name != "sqlite":
            return
        with self.engine.begin() as connection:
            columns = {
                str(row[1])
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(documents)"
                ).fetchall()
            }
            if not columns:
                return
            if "owner_session_id" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE documents "
                    "ADD COLUMN owner_session_id VARCHAR(36) "
                    "NOT NULL DEFAULT 'local'"
                )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_documents_owner_session_id "
                "ON documents (owner_session_id)"
            )

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()
