from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from finetunelab.config import RECIPE_ADAPTER, RecipeConfig
from finetunelab.devices import resolve_runtime
from finetunelab.models.registry import register_model_adapter
from finetunelab.recipes import build_trainer


class TinyTextAdapter:
    """Test adapter that exercises trainer wiring without network downloads."""

    family = "tiny-test"

    def supports(self, name_or_path: str) -> bool:
        return Path(name_or_path).name.startswith("tiny-")

    def load_processor(self, config: RecipeConfig) -> Any:
        return AutoTokenizer.from_pretrained(config.model.name_or_path)

    def load_policy_model(self, config: RecipeConfig) -> Any:
        return AutoModelForCausalLM.from_pretrained(config.model.name_or_path)

    def load_named_policy_model(self, config: RecipeConfig, name_or_path: str) -> Any:
        del config
        return AutoModelForCausalLM.from_pretrained(name_or_path)

    def load_reward_model(self, config: RecipeConfig, name_or_path: str) -> Any:
        del config
        return AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1)

    def describe_components(self, model: Any) -> dict[str, list[str]]:
        del model
        return {"language": [], "vision": [], "multimodal_projection": [], "output": []}


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "tiny-Base"
    path.mkdir()
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "user": 3,
        "assistant": 4,
        ":": 5,
        "one": 6,
        "two": 7,
        "good": 8,
        "bad": 9,
        "answer": 10,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' }}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% else %}{{ message['content'] }}{% endif %}"
        "{{ eos_token }}{% endfor %}"
    )
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.config.finetunelab_base_checkpoint = "Qwen/Qwen3.5-2B-Base"
    model.save_pretrained(path)
    register_model_adapter(TinyTextAdapter())
    return path


def _base_config(method: str, checkpoint: Path, output: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": method,
        "model": {"name_or_path": str(checkpoint), "dtype": "float32"},
        "data": {"source": str(checkpoint / "unused.jsonl"), "max_length": 24},
        "tuning": {"strategy": "full"},
        "training": {
            "output_dir": str(output),
            "max_steps": 1,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "bf16": False,
            "fp16": False,
            "tf32": False,
            "logging_steps": 1,
            "save_steps": 10,
            "report_to": ["none"],
        },
        "generation": {"max_new_tokens": 2, "num_candidates": 2},
    }


@pytest.mark.integration
@pytest.mark.parametrize("method", ["dapt", "sft", "dpo", "reward"])
def test_offline_trainers_run_one_step(method: str, tiny_checkpoint: Path, tmp_path: Path) -> None:
    raw = _base_config(method, tiny_checkpoint, tmp_path / method)
    if method == "dapt":
        rows = [{"text": "one two"}, {"text": "two one"}]
    elif method == "sft":
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ]
            }
        ] * 2
    else:
        rows = [
            {"prompt": "one", "chosen": "good answer", "rejected": "bad answer"},
            {"prompt": "two", "chosen": "good", "rejected": "bad"},
        ]
    if method == "reward":
        raw["model"]["reward_name_or_path"] = str(tiny_checkpoint)
    config = RECIPE_ADAPTER.validate_python(raw)
    trainer = build_trainer(config, Dataset.from_list(rows))
    expected_device = resolve_runtime(config).device
    assert trainer.accelerator.device.type == expected_device
    for name in ("model", "ref_model", "teacher_model"):
        auxiliary = getattr(trainer, name, None)
        if auxiliary is not None:
            assert next(auxiliary.parameters()).device.type == expected_device
    result = trainer.train()
    assert result.metrics["train_loss"] >= 0


@pytest.mark.integration
def test_grpo_trainer_runs_one_step_offline(tiny_checkpoint: Path, tmp_path: Path) -> None:
    raw = _base_config("grpo", tiny_checkpoint, tmp_path / "grpo")
    raw.update(
        {
            "rewards": [{"name": "exact_match"}],
            "num_generations": 2,
            "beta": 0.0,
        }
    )
    config = RECIPE_ADAPTER.validate_python(raw)
    dataset = Dataset.from_list(
        [{"prompt": "one", "answer": "two"}, {"prompt": "two", "answer": "one"}]
    )
    trainer = build_trainer(config, dataset)
    expected_device = resolve_runtime(config).device
    assert trainer.accelerator.device.type == expected_device
    for name in ("model", "ref_model", "teacher_model"):
        auxiliary = getattr(trainer, name, None)
        if auxiliary is not None:
            assert next(auxiliary.parameters()).device.type == expected_device
    result = trainer.train()
    assert result.metrics["train_loss"] >= 0


@pytest.mark.integration
def test_plain_prompt_completion_sft_runs_one_step(tiny_checkpoint: Path, tmp_path: Path) -> None:
    raw = _base_config("sft", tiny_checkpoint, tmp_path / "prompt_completion")
    config = RECIPE_ADAPTER.validate_python(raw)
    dataset = Dataset.from_list(
        [
            {"prompt": "one", "completion": "good answer"},
            {"prompt": "two", "completion": "good"},
        ]
    )
    trainer = build_trainer(config, dataset)
    assert trainer.args.completion_only_loss is not False
    assert not trainer.args.assistant_only_loss
    assert trainer.train().metrics["train_loss"] >= 0


@pytest.mark.integration
def test_distillation_trainer_runs_one_step_offline(tiny_checkpoint: Path, tmp_path: Path) -> None:
    raw = _base_config("distillation", tiny_checkpoint, tmp_path / "distillation")
    raw["model"]["teacher_name_or_path"] = str(tiny_checkpoint)
    config = RECIPE_ADAPTER.validate_python(raw)
    dataset = Dataset.from_list([{"prompt": "one"}, {"prompt": "two"}])
    trainer = build_trainer(config, dataset)
    expected_device = resolve_runtime(config).device
    assert trainer.accelerator.device.type == expected_device
    for name in ("model", "ref_model", "teacher_model"):
        auxiliary = getattr(trainer, name, None)
        if auxiliary is not None:
            assert next(auxiliary.parameters()).device.type == expected_device
    result = trainer.train()
    assert result.metrics["train_loss"] >= 0
