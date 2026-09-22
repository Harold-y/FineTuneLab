from pathlib import Path

import pytest

from finetunelab.config import Method, load_config
from finetunelab.errors import ConfigurationError

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "configs/qwen35/2b/dapt_lora.yaml",
        "configs/qwen35/2b/sft_qlora.yaml",
        "configs/qwen35/2b/dpo_lora.yaml",
        "configs/qwen35/2b/reward_lora.yaml",
        "configs/qwen35/2b/ppo_lora.yaml",
        "configs/qwen35/2b/grpo_qlora.yaml",
        "configs/qwen35/2b/distillation_qlora.yaml",
        "configs/qwen35/2b/image_sft_lora.yaml",
        "configs/qwen35/4b/dapt_selective.yaml",
        "configs/qwen35/4b/sft_lora.yaml",
    ],
)
def test_example_configs_parse(relative: str) -> None:
    assert load_config(ROOT / relative).schema_version == 1


def test_discriminator_selects_recipe() -> None:
    assert load_config(ROOT / "configs/qwen35/2b/dpo_lora.yaml").method == Method.DPO


def test_multimodal_ppo_fails_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: 1
method: ppo
model:
  name_or_path: Qwen/Qwen3.5-2B
data:
  source: examples/data/prompts.jsonl
  modality: image_text
  max_length: null
tuning:
  strategy: lora
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="text-only"):
        load_config(path)


def test_qlora_reward_fails_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: 1
method: reward
model:
  name_or_path: Qwen/Qwen3.5-2B
data:
  source: examples/data/preferences.jsonl
tuning:
  strategy: qlora
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="QLoRA"):
        load_config(path)
