"""Reproducibility, environment inspection, and run artifact helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from finetunelab import __version__
from finetunelab.config import RecipeConfig, dump_config, redacted_dict

TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_report() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "finetunelab": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": packages,
    }


def prepare_run_directory(config: RecipeConfig) -> Path:
    output = config.training.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dump_config(config, output / "resolved_config.yaml")
    return output


def write_manifest(
    output: Path,
    config: RecipeConfig,
    *,
    dataset_id: str | None = None,
    parameters: dict[str, Any] | None = None,
    status: str = "initialized",
) -> Path:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config": redacted_dict(config),
        "environment": environment_report(),
        "dataset_fingerprint": dataset_id,
        "parameters": parameters,
        "process": {"pid": os.getpid()},
    }
    path = output / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_manifest(path: Path, **updates: Any) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
