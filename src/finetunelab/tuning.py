"""Full, selective, LoRA, and QLoRA parameter selection."""

from __future__ import annotations

import re
from typing import Any

from finetunelab.config import RecipeConfig, TuningStrategy
from finetunelab.errors import ConfigurationError, DependencyError


def _layer_count(model: Any) -> int:
    config = getattr(model.config, "text_config", model.config)
    count = getattr(config, "num_hidden_layers", None)
    if count is None:
        raise ConfigurationError("Model config does not expose num_hidden_layers")
    return int(count)


def _selective_patterns(model: Any, config: RecipeConfig) -> list[str]:
    tuning = config.tuning
    if tuning.selective_preset == "custom":
        if not tuning.include_patterns:
            raise ConfigurationError("selective preset 'custom' requires include_patterns")
        return tuning.include_patterns
    if tuning.selective_preset == "norms_and_head":
        return [r"(?:^|\.)(?:norm|lm_head)(?:\.|$)"]
    count = _layer_count(model)
    first = max(0, count - tuning.last_n_layers)
    layer_numbers = "|".join(str(index) for index in range(first, count))
    return [rf"(?:layers|layer)\.(?:{layer_numbers})(?:\.|$)", r"(?:norm|lm_head)"]


def apply_tuning_strategy(model: Any, config: RecipeConfig) -> Any:
    """Mutate trainability or wrap the model with PEFT, then return the trainable model."""

    strategy = config.tuning.strategy
    if strategy == TuningStrategy.FULL:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    if strategy == TuningStrategy.SELECTIVE:
        includes = [re.compile(item) for item in _selective_patterns(model, config)]
        excludes = [re.compile(item) for item in config.tuning.exclude_patterns]
        selected = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(pattern.search(name) for pattern in includes) and not any(
                pattern.search(name) for pattern in excludes
            )
            selected += int(parameter.requires_grad)
        if selected == 0:
            raise ConfigurationError("Selective tuning patterns matched no parameters")
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise DependencyError("LoRA requires PEFT") from exc

    if strategy == TuningStrategy.QLORA:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )
    exclude_expression = (
        ".*(?:" + "|".join(config.tuning.exclude_patterns) + ").*"
        if config.tuning.exclude_patterns
        else None
    )
    task_type = TaskType.SEQ_CLS if str(config.method) == "reward" else TaskType.CAUSAL_LM
    lora = LoraConfig(
        r=config.tuning.lora_rank,
        lora_alpha=config.tuning.lora_alpha,
        lora_dropout=config.tuning.lora_dropout,
        target_modules=config.tuning.lora_target_modules,
        exclude_modules=exclude_expression,
        modules_to_save=config.tuning.modules_to_save or None,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, lora)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model


def parameter_report(model: Any) -> dict[str, int | float]:
    """Return exact trainable and total parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }
