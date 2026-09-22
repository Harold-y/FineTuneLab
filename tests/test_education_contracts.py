"""Regression tests for the interfaces exercised by the course."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from finetunelab.config import RECIPE_ADAPTER
from finetunelab.data.loader import _local_builder, load_datasets
from finetunelab.education import inspect_local_checkpoint
from finetunelab.errors import ConfigurationError, DataValidationError
from finetunelab.models.qwen35 import Qwen35Adapter
from finetunelab.workflows import evaluate


def recipe(tmp_path: Path, method: str = "sft") -> dict:
    return {
        "method": method,
        "model": {"name_or_path": "Qwen/Qwen3.5-2B-Base", "local_files_only": True},
        "data": {"source": str(tmp_path), "source_type": "local"},
        "training": {"output_dir": str(tmp_path / "run")},
    }


def test_local_split_mapping_and_missing_split_diagnostics(tmp_path: Path) -> None:
    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.jsonl").write_text('{"text":"example"}\n', encoding="utf-8")
    raw = recipe(tmp_path, "dapt")
    raw["data"].update(
        data_files={name: f"{name}.jsonl" for name in ("train", "validation", "test")},
        eval_split="validation",
    )
    config = RECIPE_ADAPTER.validate_python(raw)
    builder, files = _local_builder(config)
    assert builder == "json"
    assert Path(files["test"]) == tmp_path / "test.jsonl"
    rows = Dataset.from_list([{"text": "example"}])
    with (
        patch("finetunelab.data.loader.load_dataset", return_value={"train": rows}),
        pytest.raises(DataValidationError, match="Missing evaluation split"),
    ):
        load_datasets(config)
    with patch(
        "finetunelab.data.loader.load_dataset", return_value={"train": rows, "validation": rows}
    ):
        train, validation = load_datasets(config)
        assert len(train) == len(validation) == 1


def test_local_model_recognition_is_json_based(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    assert Qwen35Adapter().supports(str(tmp_path))
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    assert not Qwen35Adapter().supports(str(tmp_path))


def test_renamed_dapt_snapshot_requires_provenance(tmp_path: Path) -> None:
    raw = recipe(tmp_path, "dapt")
    raw["model"]["name_or_path"] = str(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Base checkpoint identity"):
        RECIPE_ADAPTER.validate_python(raw)
    metadata = {
        "model_type": "qwen3_5",
        "finetunelab_base_checkpoint": "Qwen/Qwen3.5-4B-Base",
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert RECIPE_ADAPTER.validate_python(raw).model.local_files_only


def test_offline_flag_reaches_processor_and_model_loaders(tmp_path: Path) -> None:
    config = RECIPE_ADAPTER.validate_python(recipe(tmp_path))
    adapter = Qwen35Adapter()
    assert adapter._model_kwargs(config)["local_files_only"] is True
    with patch("transformers.AutoProcessor.from_pretrained") as loader:
        adapter.load_processor(config)
        assert loader.call_args.kwargs["local_files_only"] is True


def test_evaluation_never_silently_falls_back_to_base(tmp_path: Path) -> None:
    config = RECIPE_ADAPTER.validate_python(recipe(tmp_path))
    rows = Dataset.from_list([{"text": "example"}])
    with (
        patch("finetunelab.workflows.load_datasets", return_value=(rows, rows)),
        patch("finetunelab.workflows.build_trainer") as factory,
    ):
        with pytest.raises(ConfigurationError, match="No trained artifact"):
            evaluate(config)
        factory.assert_not_called()


def test_incomplete_weight_shards_are_diagnosed(tmp_path: Path) -> None:
    for filename in ("config.json", "tokenizer_config.json"):
        (tmp_path / filename).write_text("{}", encoding="utf-8")
    (tmp_path / "first.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Missing weight shards"):
        inspect_local_checkpoint(tmp_path)


def test_reward_evaluation_does_not_require_vision_processor(tmp_path: Path) -> None:
    raw = recipe(tmp_path, "reward")
    config = RECIPE_ADAPTER.validate_python(raw)
    rows = Dataset.from_list([{"prompt": "one", "chosen": "yes", "rejected": "no"}])
    checkpoint = tmp_path / "saved_reward"
    checkpoint.mkdir()
    trainer = SimpleNamespace(model=torch.nn.Linear(2, 1), evaluate=lambda: {"eval_accuracy": 1.0})
    with (
        patch("finetunelab.workflows.load_datasets", return_value=(rows, rows)),
        patch("finetunelab.workflows.build_trainer", return_value=trainer),
        patch("finetunelab.workflows.get_model_adapter") as adapter,
    ):
        metrics = evaluate(config, checkpoint=str(checkpoint))
        assert metrics["checkpoint"] == str(checkpoint)
        adapter.assert_not_called()
    sample_file = next(config.training.output_dir.glob("evaluations/*/generated_samples.json"))
    assert json.loads(sample_file.read_text(encoding="utf-8")) == []


def test_ppo_policy_adapter_is_not_loaded_into_reward_model(tmp_path: Path) -> None:
    raw = recipe(tmp_path, "ppo")
    raw["model"]["adapter_name_or_path"] = str(tmp_path / "policy_adapter")
    config = RECIPE_ADAPTER.validate_python(raw)
    reward_model = SimpleNamespace(config=SimpleNamespace(pad_token_id=0))
    with (
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=reward_model,
        ),
        patch("peft.PeftModel.from_pretrained") as peft_loader,
    ):
        assert Qwen35Adapter().load_reward_model(config, str(tmp_path)) is reward_model
        peft_loader.assert_not_called()
