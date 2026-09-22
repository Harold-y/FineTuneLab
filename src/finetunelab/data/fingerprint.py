"""Dataset identity helpers for reproducible run manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from finetunelab.config import RecipeConfig


def dataset_fingerprint(dataset: Any, config: RecipeConfig) -> str:
    native = getattr(dataset, "_fingerprint", None)
    if native:
        return str(native)
    payload = {
        "source": config.data.source,
        "split": config.data.train_split,
        "streaming": config.data.streaming,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
