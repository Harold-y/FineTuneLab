"""Canonical schema checks performed before model allocation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from finetunelab.config import Method, Modality, RecipeConfig
from finetunelab.errors import DataValidationError

REQUIRED_ALTERNATIVES: dict[Method, tuple[set[str], ...]] = {
    Method.DAPT: ({"text"}, {"messages"}),
    Method.SFT: ({"messages"}, {"prompt", "completion"}),
    Method.DPO: ({"prompt", "chosen", "rejected"},),
    Method.REWARD: ({"prompt", "chosen", "rejected"},),
    Method.PPO: ({"prompt"},),
    Method.GRPO: ({"prompt"},),
    Method.DISTILLATION: ({"prompt"},),
}


def _sample_rows(dataset: Any, limit: int = 32) -> Iterable[dict[str, Any]]:
    try:
        count = min(len(dataset), limit)
    except TypeError:
        if hasattr(dataset, "take"):
            yield from dataset.take(limit)
            return
        raise DataValidationError("Dataset is neither sized nor iterable") from None
    for index in range(count):
        yield dataset[index]


def validate_dataset(dataset: Any, config: RecipeConfig) -> dict[str, Any]:
    """Validate schema and common value errors using a bounded sample."""

    checked = 0
    for index, row in enumerate(_sample_rows(dataset)):
        checked += 1
        keys = set(row)
        alternatives = REQUIRED_ALTERNATIVES[config.method]
        if not any(required <= keys for required in alternatives):
            expected = " or ".join("{" + ", ".join(sorted(item)) + "}" for item in alternatives)
            raise DataValidationError(
                f"Row {index} does not match {config.method} schema. Expected {expected}; "
                f"found {sorted(keys)}"
            )
        if config.data.modality == Modality.IMAGE_TEXT and not ({"image", "images"} & keys):
            raise DataValidationError(f"Row {index} has no image/images field")
        if config.method in {Method.DPO, Method.REWARD} and row["chosen"] == row["rejected"]:
            raise DataValidationError(f"Row {index} has identical chosen and rejected responses")
        if "messages" in row:
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise DataValidationError(f"Row {index} messages must be a non-empty list")
            for message in messages:
                if not isinstance(message, dict) or not {"role", "content"} <= set(message):
                    raise DataValidationError(
                        f"Row {index} contains a message without role/content"
                    )
    if checked == 0:
        raise DataValidationError("Dataset is empty")
    return {"sampled_rows": checked, "method": str(config.method)}
