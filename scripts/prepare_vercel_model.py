"""Download the pinned multilingual model into the Vercel function bundle."""

from __future__ import annotations

import shutil
from pathlib import Path

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "deployment" / "model"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "1_Pooling/config.json",
)


def prepare_model(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> Path:
    """Materialize one reproducible model snapshot without committing weights."""

    target = DEFAULT_TARGET
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=target,
        allow_patterns=("*.json", "*.safetensors", "1_Pooling/*"),
    )
    missing = [relative for relative in REQUIRED_FILES if not (target / relative).is_file()]
    if missing:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError("Bundled semantic model is incomplete.")
    return target


def main() -> None:
    target = prepare_model()
    print(f"Semantic model prepared at {target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
