"""Database configuration tests that never require a live PostgreSQL server."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from app.config import _database_url_from_environment
from app.database.session import (
    Database,
    database_engine_options,
    normalize_database_url,
)


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        (
            "postgres://student@ep-example-pooler.neon.tech/neondb?sslmode=require",
            "postgresql+psycopg://student@ep-example-pooler.neon.tech/"
            "neondb?sslmode=require",
        ),
        (
            "postgresql://student@localhost/skripsicheck",
            "postgresql+psycopg://student@localhost/skripsicheck",
        ),
        (
            "postgresql+psycopg://student@localhost/skripsicheck",
            "postgresql+psycopg://student@localhost/skripsicheck",
        ),
        ("sqlite:///data/local.sqlite3", "sqlite:///data/local.sqlite3"),
    ],
)
def test_database_url_normalization(raw_url: str, expected: str) -> None:
    assert normalize_database_url(raw_url) == expected


def test_database_url_rejects_blank_value() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_database_url("  ")


def test_invalid_database_url_has_secret_safe_error() -> None:
    with pytest.raises(ValueError) as error:
        database_engine_options("not-a-database-url")

    assert str(error.value) == "Database URL is invalid."


def test_postgres_engine_options_are_serverless_safe() -> None:
    options = database_engine_options("postgresql://user:password@localhost/database")

    assert options["pool_pre_ping"] is True
    assert options["poolclass"] is NullPool
    assert "connect_args" not in options


def test_sqlite_engine_options_keep_thread_compatibility() -> None:
    options = database_engine_options("sqlite:///:memory:")

    assert options == {
        "pool_pre_ping": True,
        "connect_args": {"check_same_thread": False},
    }


def test_sqlite_database_still_creates_parent_schema_and_foreign_keys(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "skripsicheck.sqlite3"
    database = Database(f"sqlite:///{database_path.as_posix()}")
    try:
        database.create_schema()
        with database.session() as session:
            foreign_keys = session.execute(text("PRAGMA foreign_keys")).scalar_one()

        assert database_path.is_file()
        assert foreign_keys == 1
    finally:
        database.close()


def test_database_environment_prefers_explicit_project_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_URL", "postgres://legacy.example/database")
    monkeypatch.setenv("DATABASE_URL", "postgres://generic.example/database")
    monkeypatch.setenv(
        "SKRIPSICHECK_DATABASE_URL",
        "postgres://explicit.example/database",
    )

    assert _database_url_from_environment() == "postgres://explicit.example/database"


def test_database_environment_supports_neon_and_keeps_sqlite_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKRIPSICHECK_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://generic.example/database")
    monkeypatch.setenv("POSTGRES_URL", "postgres://neon.example/database")
    assert _database_url_from_environment() == "postgres://generic.example/database"

    monkeypatch.delenv("DATABASE_URL")
    assert _database_url_from_environment() == "postgres://neon.example/database"

    monkeypatch.delenv("POSTGRES_URL")
    assert _database_url_from_environment() == "sqlite:///data/skripsicheck.sqlite3"
