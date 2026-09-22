"""Registration and lookup for model-family adapters."""

from __future__ import annotations

from finetunelab.errors import ConfigurationError
from finetunelab.protocols import ModelAdapter

_ADAPTERS: list[ModelAdapter] = []


def register_model_adapter(adapter: ModelAdapter) -> None:
    """Register an adapter once, preserving lookup order."""

    if not any(type(existing) is type(adapter) for existing in _ADAPTERS):
        _ADAPTERS.append(adapter)


def get_model_adapter(name_or_path: str) -> ModelAdapter:
    """Return the first adapter that recognizes the requested checkpoint."""

    if not _ADAPTERS:
        from finetunelab.models.qwen35 import Qwen35Adapter

        register_model_adapter(Qwen35Adapter())
    for adapter in _ADAPTERS:
        if adapter.supports(name_or_path):
            return adapter
    raise ConfigurationError(f"No model adapter recognizes {name_or_path!r}")
