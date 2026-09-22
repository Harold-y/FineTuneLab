"""Composable reward functions used by GRPO and evaluation."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Callable
from typing import Any, cast

from finetunelab.config import RewardSpec
from finetunelab.errors import ConfigurationError


def completion_text(value: Any) -> str:
    """Normalize the completion shapes produced by text and conversational trainers."""

    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
    return str(value)


def _answers(kwargs: dict[str, Any], count: int) -> list[Any]:
    values = kwargs.get("answer", kwargs.get("reference_answer"))
    if values is None:
        raise ValueError("This reward requires an answer or reference_answer dataset column")
    if isinstance(values, list):
        return values
    return [values] * count


def exact_match_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    answers = _answers(kwargs, len(completions))
    return [
        float(completion_text(item).strip() == str(answer).strip())
        for item, answer in zip(completions, answers, strict=True)
    ]


def _last_number(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", ""))
    return float(matches[-1]) if matches else None


def numeric_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    answers = _answers(kwargs, len(completions))
    scores: list[float] = []
    for completion, answer in zip(completions, answers, strict=True):
        predicted = _last_number(completion_text(completion))
        expected = _last_number(str(answer))
        scores.append(
            float(
                predicted is not None and expected is not None and math.isclose(predicted, expected)
            )
        )
    return scores


def make_regex_reward(pattern: str) -> Callable[..., list[float]]:
    compiled = re.compile(pattern, re.DOTALL)

    def reward(completions: list[Any], **_: Any) -> list[float]:
        return [float(bool(compiled.search(completion_text(item)))) for item in completions]

    reward.__name__ = "regex_reward"
    return reward


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    pattern = re.compile(r"<think>.*?</think>\s*.+", re.DOTALL)
    return [float(bool(pattern.fullmatch(completion_text(item).strip()))) for item in completions]


def length_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    target = int(kwargs.get("target_length", 256))
    return [
        max(0.0, 1.0 - abs(len(completion_text(item)) - target) / max(target, 1))
        for item in completions
    ]


def _load_callable(path: str) -> Callable[..., list[float]]:
    if ":" not in path:
        raise ConfigurationError("Python rewards use the 'module:function' notation")
    module_name, function_name = path.split(":", 1)
    value = getattr(importlib.import_module(module_name), function_name)
    if not callable(value):
        raise ConfigurationError(f"Reward target is not callable: {path}")
    return cast(Callable[..., list[float]], value)


def build_reward_functions(specs: list[RewardSpec]) -> list[Callable[..., list[float]]]:
    """Build weighted TRL-compatible reward callables from configuration."""

    built: list[Callable[..., list[float]]] = []
    for spec in specs:
        if spec.name == "exact_match":
            base = exact_match_reward
        elif spec.name == "numeric":
            base = numeric_reward
        elif spec.name == "regex":
            if not spec.pattern:
                raise ConfigurationError("regex reward requires pattern")
            base = make_regex_reward(spec.pattern)
        elif spec.name == "format":
            base = format_reward
        elif spec.name == "length":
            base = length_reward
        elif spec.name == "python":
            if not spec.callable:
                raise ConfigurationError("python reward requires callable")
            base = _load_callable(spec.callable)
        else:  # pragma: no cover - Pydantic prevents this
            raise ConfigurationError(f"Unknown reward: {spec.name}")

        def weighted(
            completions: list[Any],
            _function: Callable[..., list[float]] = base,
            _spec: RewardSpec = spec,
            **kwargs: Any,
        ) -> list[float]:
            values = _function(completions, **kwargs)
            if len(values) != len(completions):
                raise ValueError("Reward function returned the wrong number of scores")
            result = [float(value) * _spec.weight for value in values]
            if not all(math.isfinite(value) for value in result):
                raise ValueError("Reward function returned NaN or infinity")
            return result

        weighted.__name__ = f"weighted_{spec.name}"
        built.append(weighted)
    return built
