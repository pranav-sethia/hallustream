"""Shared Modal setup: one image built from the locked dependencies, one Volume for caches."""

import json
import platform
import subprocess
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = "/cache"  # Modal Volume mount point (features, results, HF cache)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(PROJECT_ROOT), extras=["research"], frozen=True)
    .env({"HF_HOME": f"{CACHE_DIR}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("hallustream")
)

volume = modal.Volume.from_name("hallustream-cache", create_if_missing=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_run_meta(out_dir: Path, *, job: str, commit: str, config: dict, extra: dict) -> None:
    """Record what produced a result, so every number can be traced back."""
    import torch
    import transformers

    meta = {
        "job": job,
        "git_commit": commit,
        "config": config,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
