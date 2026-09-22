"""Shared backend selection and setup; no training objectives live here."""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch

from finetunelab.config import DeviceChoice, RecipeConfig
from finetunelab.errors import ConfigurationError

TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}


def select_device(configured: str | None = None, override: str | None = None) -> tuple[str, str]:
    requested = override if override is not None else configured
    if requested is None:
        requested = os.environ.get("FTLAB_DEVICE", "auto")
    if requested not in set(DeviceChoice):
        raise ConfigurationError(f"Unknown device {requested!r}; choose auto, cpu, mps, or cuda")
    device = requested
    if device == "auto":
        if os.environ.get("ACCELERATE_USE_CPU", "").lower() in TRUE_VALUES:
            device = "cpu"
        elif torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("CUDA is unavailable. Select cpu or an available mps backend.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ConfigurationError("MPS is unavailable. Select cpu or an available cuda backend.")
    return requested, device


@dataclass(frozen=True)
class DeviceRuntime:
    requested: str
    device: str
    dtype: str
    attention: str
    bf16: bool = False
    fp16: bool = False
    tf32: bool = False

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)  # type: ignore[no-any-return]

    def report(self) -> dict[str, Any]:
        return asdict(self)

    def precision_context(self) -> Any:
        if self.device == "cuda" and (self.bf16 or self.fp16):
            dtype = torch.bfloat16 if self.bf16 else torch.float16
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()


def resolve_runtime(
    config: RecipeConfig | None = None,
    *,
    device: str | None = None,
    dtype: str | None = None,
) -> DeviceRuntime:
    configured = None
    if config is not None and "device" in config.training.model_fields_set:
        configured = config.training.device
    requested, backend = select_device(configured, device)
    if dtype is None and config is not None and "dtype" in config.model.model_fields_set:
        dtype = config.model.dtype
    if dtype is None or dtype == "auto":
        dtype = "float32" if backend == "cpu" else "bfloat16"
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ConfigurationError(f"Unsupported model dtype: {dtype}")
    if backend == "cpu" and dtype == "float16":
        raise ConfigurationError("CPU float16 training is not supported; use float32 or bfloat16")
    if backend == "mps" and dtype == "bfloat16" and not torch.backends.mps.is_macos_or_newer(14, 0):
        raise ConfigurationError("MPS bfloat16 requires macOS 14 or newer; select float32")
    if backend == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ConfigurationError("This CUDA device lacks BF16 support; select model.dtype=float16")
    attention = "eager" if backend != "cuda" else "sdpa"
    flags = {"bf16": False, "fp16": False, "tf32": False}
    if backend == "cuda":
        flags = {"bf16": dtype == "bfloat16", "fp16": dtype == "float16", "tf32": True}
    if config is not None:
        if "attention_implementation" in config.model.model_fields_set:
            attention = config.model.attention_implementation
        if attention == "flash_attention_2" and backend != "cuda":
            raise ConfigurationError("flash_attention_2 requires CUDA; use eager or sdpa")
        if config.tuning.strategy == "qlora" and backend != "cuda":
            raise ConfigurationError("This CPU/MPS profile supports LoRA, not QLoRA; select lora")
        if backend != "cuda" and (config.training.deepspeed or config.training.fsdp):
            raise ConfigurationError("DeepSpeed/FSDP are supported only by the CUDA profile")
        for name in flags:
            if name in config.training.model_fields_set:
                value = bool(getattr(config.training, name))
                if backend != "cuda" and value:
                    raise ConfigurationError(
                        f"training.{name}=true is unsupported in the {backend} profile; "
                        "omit it or set false. Model weight dtype is configured separately."
                    )
                flags[name] = value
    return DeviceRuntime(requested, backend, dtype, attention, **flags)


def activate_runtime(runtime: DeviceRuntime) -> None:
    """Select Accelerate's backend before initialization without resetting shared state."""
    from accelerate.state import AcceleratorState, PartialState

    for state in (PartialState._shared_state, AcceleratorState._shared_state):
        active = state.get("device")
        if active is not None and torch.device(active).type != runtime.device:
            raise ConfigurationError(
                f"Accelerate is already initialized on {active}; requested {runtime.device}. "
                "Restart the notebook kernel or use a fresh process before changing devices."
            )
    os.environ["ACCELERATE_USE_CPU"] = str(runtime.device == "cpu").lower()
    # Let distributed CUDA select the rank-local GPU instead of pinning every rank.
    if runtime.device == "cuda":
        os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count())
    else:
        os.environ["ACCELERATE_TORCH_DEVICE"] = runtime.device
    if runtime.device == "mps":
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"


def empty_device_cache(device: str | torch.device) -> None:
    gc.collect()
    backend = torch.device(device).type
    if backend == "mps":
        torch.mps.empty_cache()
    elif backend == "cuda":
        torch.cuda.empty_cache()


def device_rng_state(device: str | torch.device) -> Any:
    backend = torch.device(device).type
    if backend == "mps":
        return torch.mps.get_rng_state()
    if backend == "cuda":
        return torch.cuda.get_rng_state_all()
    return None


def restore_device_rng_state(device: str | torch.device, state: Any) -> None:
    backend = torch.device(device).type
    if backend == "mps" and state is not None:
        torch.mps.set_rng_state(state.cpu())
    elif backend == "cuda" and state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state])
