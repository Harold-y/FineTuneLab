"""Backend selection must be explicit, consistent, and fail before loading weights."""

from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from finetunelab.cli import app
from finetunelab.config import RECIPE_ADAPTER
from finetunelab.devices import activate_runtime, resolve_runtime, select_device
from finetunelab.errors import ConfigurationError
from finetunelab.recipes.factory import _common_args
from finetunelab.runtime import prepare_run_directory


def recipe():
    return RECIPE_ADAPTER.validate_python(
        {
            "method": "sft",
            "model": {"name_or_path": "Qwen/Qwen3.5-2B"},
            "data": {"source": "examples/data/sft.jsonl"},
        }
    )


def test_device_precedence(monkeypatch):
    monkeypatch.setenv("FTLAB_DEVICE", "cuda")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    with patch("torch.cuda.is_available", return_value=True):
        assert select_device() == ("cuda", "cuda")
        assert select_device("cpu") == ("cpu", "cpu")
        assert select_device("cuda", "cpu") == ("cpu", "cpu")
        assert select_device("auto") == ("auto", "cpu")


def test_automatic_device_priority(monkeypatch):
    monkeypatch.delenv("FTLAB_DEVICE", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_CPU", raising=False)
    with patch("torch.cuda.is_available", return_value=True):
        assert select_device()[1] == "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=True):
            assert select_device()[1] == "mps"
        with patch("torch.backends.mps.is_available", return_value=False):
            assert select_device()[1] == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_explicit_device_never_falls_back(device):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        pytest.raises(ConfigurationError, match="unavailable"),
    ):
        select_device(device)


def test_cpu_defaults_and_explicit_precision(monkeypatch):
    monkeypatch.setenv("FTLAB_DEVICE", "cpu")
    config = recipe()
    runtime = resolve_runtime(config)
    assert runtime.dtype == "float32" and runtime.attention == "eager"
    assert not runtime.bf16 and not runtime.fp16 and not runtime.tf32
    assert _common_args(config)["use_cpu"] is True
    config.training.bf16 = True
    with pytest.raises(ConfigurationError, match="training.bf16"):
        resolve_runtime(config)


@pytest.mark.parametrize("option", ["qlora", "flash", "distributed"])
def test_incompatible_profiles_fail_early(option):
    config = recipe()
    config.training.device = "cpu"
    if option == "qlora":
        config.tuning.strategy = "qlora"
    elif option == "flash":
        config.model.attention_implementation = "flash_attention_2"
    else:
        config.training.deepspeed = "unused.json"
    with pytest.raises(ConfigurationError):
        resolve_runtime(config)


def test_initialized_accelerate_requires_restart(monkeypatch):
    from accelerate.state import AcceleratorState, PartialState

    runtime = resolve_runtime(device="cpu")
    monkeypatch.setattr(PartialState, "_shared_state", {"device": torch.device("mps")})
    monkeypatch.setattr(AcceleratorState, "_shared_state", {})
    with pytest.raises(ConfigurationError, match="Restart"):
        activate_runtime(runtime)


def test_explicit_mps_overrides_mac_kernel_cpu_setting(monkeypatch):
    from accelerate.state import AcceleratorState, PartialState

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")
    monkeypatch.setattr(PartialState, "_shared_state", {})
    monkeypatch.setattr(AcceleratorState, "_shared_state", {})
    with patch("torch.backends.mps.is_available", return_value=True):
        runtime = resolve_runtime(device="mps", dtype="float32")
    activate_runtime(runtime)
    import os

    assert os.environ["ACCELERATE_USE_CPU"] == "false"
    assert os.environ["ACCELERATE_TORCH_DEVICE"] == "mps"
    assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"


def test_cli_device_overrides_yaml():
    with patch("finetunelab.cli.train", return_value={}) as train:
        result = CliRunner().invoke(
            app, ["train", "-c", "configs/qwen35/4b/sft_lora.yaml", "--device", "cpu"]
        )
    assert result.exit_code == 0, result.output
    assert train.call_args.args[0].training.device == "cpu"


def test_resolved_config_can_be_replayed(tmp_path):
    from finetunelab.config import load_config

    config = recipe()
    config.training.output_dir = tmp_path
    config.training.device = "cpu"
    runtime = resolve_runtime(config)
    prepare_run_directory(config, runtime)
    reloaded = load_config(tmp_path / "resolved_config.yaml")
    assert resolve_runtime(reloaded) == runtime
