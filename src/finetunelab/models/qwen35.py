"""Qwen3.5 loading and component discovery.

The adapter intentionally contains all architecture-specific assumptions. Training
recipes operate only on the public adapter interface, which makes adding another
model family a contained change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from finetunelab.config import SUPPORTED_QWEN35_MODELS, RecipeConfig, TuningStrategy
from finetunelab.devices import activate_runtime, resolve_runtime
from finetunelab.errors import DependencyError, FineTuneLabError


def _dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16


class Qwen35Adapter:
    """Adapter for the dense Qwen3.5 2B and 4B multimodal checkpoints."""

    family = "qwen3.5"

    def supports(self, name_or_path: str) -> bool:
        if name_or_path in SUPPORTED_QWEN35_MODELS:
            return True
        path = Path(name_or_path)
        if not path.exists():
            return False
        config_path = path / "config.json"
        if not config_path.exists():
            return False
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            return bool(metadata.get("model_type") == "qwen3_5")
        except (ValueError, OSError):
            return False

    def quantization_config(self, config: RecipeConfig) -> Any | None:
        """Build the BitsAndBytes configuration only for QLoRA."""

        if config.tuning.strategy != TuningStrategy.QLORA:
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise DependencyError(
                "QLoRA requires the 'quant' extra: uv sync --extra quant"
            ) from exc
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.tuning.quant_type,
            bnb_4bit_use_double_quant=config.tuning.double_quant,
            bnb_4bit_compute_dtype=_dtype(config.tuning.quant_compute_dtype),
        )

    def load_processor(self, config: RecipeConfig) -> Any:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            config.model.name_or_path,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=config.model.local_files_only,
        )

    def _model_kwargs(self, config: RecipeConfig) -> dict[str, Any]:
        runtime = resolve_runtime(config)
        activate_runtime(runtime)
        kwargs: dict[str, Any] = {
            "revision": config.model.revision,
            "trust_remote_code": config.model.trust_remote_code,
            "local_files_only": config.model.local_files_only,
            "dtype": runtime.torch_dtype,
        }
        if runtime.attention != "auto":
            kwargs["attn_implementation"] = runtime.attention
        if runtime.device != "cuda":
            kwargs["device_map"] = {"": runtime.device}
        elif not config.training.deepspeed and not config.training.fsdp:
            kwargs["device_map"] = {"": torch.cuda.current_device()}
        quantization = self.quantization_config(config)
        if quantization is not None:
            kwargs["quantization_config"] = quantization
            kwargs["device_map"] = {"": torch.cuda.current_device()}
        return kwargs

    def load_policy_model(self, config: RecipeConfig) -> Any:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            config.model.name_or_path, **self._model_kwargs(config)
        )
        if config.model.base_checkpoint:
            model.config.finetunelab_base_checkpoint = config.model.base_checkpoint
        if config.model.adapter_name_or_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                config.model.adapter_name_or_path,
                local_files_only=config.model.local_files_only,
            )
        if config.training.gradient_checkpointing:
            model.config.use_cache = False
        return model

    def load_named_policy_model(self, config: RecipeConfig, name_or_path: str) -> Any:
        """Load a reference/teacher policy without mutating the main model config."""

        from transformers import AutoModelForImageTextToText

        kwargs = self._model_kwargs(config)
        # A reference or teacher must not inherit the student's 4-bit setting implicitly.
        kwargs.pop("quantization_config", None)
        return AutoModelForImageTextToText.from_pretrained(name_or_path, **kwargs)

    def load_reward_model(self, config: RecipeConfig, name_or_path: str) -> Any:
        from transformers import AutoModelForSequenceClassification

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                name_or_path,
                **self._model_kwargs(config),
                num_labels=1,
            )
            if getattr(model.config, "pad_token_id", None) is None:
                text_config = getattr(model.config, "text_config", model.config)
                model.config.pad_token_id = text_config.pad_token_id
            # A PPO policy adapter must not be attached to its reward/value models.
            if config.model.adapter_name_or_path and str(config.method) == "reward":
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    config.model.adapter_name_or_path,
                    local_files_only=config.model.local_files_only,
                )
            return model
        except (ValueError, OSError) as exc:
            raise FineTuneLabError(
                "The selected reward/value checkpoint cannot be loaded as a sequence "
                "classifier. Provide model.reward_name_or_path/value_name_or_path pointing "
                "to a compatible text reward model."
            ) from exc

    def describe_components(self, model: Any) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "language": [],
            "vision": [],
            "multimodal_projection": [],
            "output": [],
        }
        for name, _ in model.named_modules():
            lowered = name.lower()
            if any(token in lowered for token in ("merger", "projector", "multimodal")):
                groups["multimodal_projection"].append(name)
            elif any(token in lowered for token in ("visual", "vision")):
                groups["vision"].append(name)
            elif name.endswith(("lm_head", "norm")):
                groups["output"].append(name)
            elif any(token in lowered for token in ("language_model", "model.layers")):
                groups["language"].append(name)
        return groups
