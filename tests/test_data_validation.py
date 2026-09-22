from pathlib import Path

import pytest

from finetunelab.config import load_config
from finetunelab.data.validation import validate_dataset
from finetunelab.errors import DataValidationError

ROOT = Path(__file__).parents[1]


def test_sft_conversation_schema() -> None:
    config = load_config(ROOT / "configs/qwen35/2b/sft_qlora.yaml")
    rows = [{"messages": [{"role": "user", "content": "Hi"}]}]
    assert validate_dataset(rows, config)["sampled_rows"] == 1


def test_identical_preferences_are_rejected() -> None:
    config = load_config(ROOT / "configs/qwen35/2b/dpo_lora.yaml")
    rows = [{"prompt": "p", "chosen": "same", "rejected": "same"}]
    with pytest.raises(DataValidationError, match="identical"):
        validate_dataset(rows, config)


def test_empty_dataset_is_rejected() -> None:
    config = load_config(ROOT / "configs/qwen35/2b/sft_qlora.yaml")
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataset([], config)
