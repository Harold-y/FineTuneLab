"""Model adapter registry."""

from finetunelab.models.qwen35 import Qwen35Adapter
from finetunelab.models.registry import get_model_adapter

__all__ = ["Qwen35Adapter", "get_model_adapter"]
