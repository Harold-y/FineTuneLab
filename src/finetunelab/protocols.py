"""Small interfaces that keep model, data, reward, and trainer code replaceable."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from finetunelab.config import RecipeConfig


@runtime_checkable
class ModelAdapter(Protocol):
    family: str

    def supports(self, name_or_path: str) -> bool: ...

    def load_processor(self, config: RecipeConfig) -> Any: ...

    def load_policy_model(self, config: RecipeConfig) -> Any: ...

    def load_reward_model(self, config: RecipeConfig, name_or_path: str) -> Any: ...

    def describe_components(self, model: Any) -> dict[str, list[str]]: ...


@runtime_checkable
class DatasetAdapter(Protocol):
    def load(self, config: RecipeConfig) -> tuple[Any, Any | None]: ...


@runtime_checkable
class TrainingRecipe(Protocol):
    method: str

    def build(self, config: RecipeConfig, dataset: Any, eval_dataset: Any | None) -> Any: ...


@runtime_checkable
class RewardFunction(Protocol):
    name: str

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]: ...


@runtime_checkable
class Judge(Protocol):
    def compare(self, prompt: str, candidates: list[str]) -> dict[str, Any]: ...
