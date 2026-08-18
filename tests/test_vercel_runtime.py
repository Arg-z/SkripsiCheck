"""Static checks for the Vercel FastAPI runtime contract."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from app.config import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_python_runtime_is_pinned_to_supported_312() -> None:
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_fastapi_entrypoint_is_explicit() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["vercel"]["entrypoint"] == "app.main:app"


def test_vercel_function_uses_fluid_compute_in_singapore() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert config["$schema"] == "https://openapi.vercel.sh/vercel.json"
    assert config["framework"] == "fastapi"
    assert config["fluid"] is True
    assert config["regions"] == ["sin1"]

    function = config["functions"]["app/main.py"]
    assert function["maxDuration"] == 300
    assert "memory" not in function
    assert "runtime" not in function
    assert "builds" not in config
    assert "routes" not in config


def test_development_artifacts_are_excluded_from_function_bundle() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    excluded = config["functions"]["app/main.py"]["excludeFiles"]

    for pattern in ("tests/**", "scripts/**", "sample_documents/**", "data/**"):
        assert pattern in excluded

    # The browser UI is served by FastAPI, so these must stay bundled.
    assert "static/**" not in excluded
    assert "app/templates/**" not in excluded


def test_production_requirements_exclude_test_and_local_server_tools() -> None:
    production = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    development = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").lower()

    assert "pytest" not in production
    assert "httpx" not in production
    assert "uvicorn" not in production
    assert "pytest" in development
    assert "httpx" in development
    assert "uvicorn" in development
    # The Blob adapter is verified against this concrete pre-1.0 SDK contract.
    assert "vercel==0.10.0" in production


def test_local_vercel_link_metadata_is_not_committed() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".vercel/" in ignored.splitlines()


def test_browser_blob_prefix_is_fixed_until_all_runtimes_can_share_config() -> None:
    valid = Settings(
        storage_backend="vercel_blob",
        blob_document_prefix="documents",
        access_token="a-secure-test-token-with-at-least-32-chars",
    )
    valid.validate()

    invalid = Settings(
        storage_backend="vercel_blob",
        blob_document_prefix="custom-documents",
        access_token="a-secure-test-token-with-at-least-32-chars",
    )
    with pytest.raises(ValueError, match="PREFIX=documents"):
        invalid.validate()


def test_access_token_must_fit_browser_and_node_payload_contract() -> None:
    Settings(access_token="x" * 32).validate()
    Settings(access_token="x" * 512).validate()

    with pytest.raises(ValueError, match="between 32 and 512"):
        Settings(access_token="x" * 31).validate()
    with pytest.raises(ValueError, match="between 32 and 512"):
        Settings(access_token="x" * 513).validate()
