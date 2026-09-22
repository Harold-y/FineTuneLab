"""Trainer factories for every FineTuneLab method.

This module keeps third-party trainer APIs at the boundary. The rest of the
framework deals in stable FineTuneLab configuration objects.
"""

from __future__ import annotations

import inspect
from typing import Any

from finetunelab.config import (
    DAPTConfig,
    DistillationConfig,
    DPOConfig,
    GRPOConfig,
    Method,
    Modality,
    PPOConfig,
    RecipeConfig,
    RewardConfig,
    SFTConfig,
)
from finetunelab.devices import activate_runtime, resolve_runtime
from finetunelab.errors import FineTuneLabError
from finetunelab.models import get_model_adapter
from finetunelab.rewards import build_reward_functions
from finetunelab.tuning import apply_tuning_strategy


def _supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Filter optional arguments across nearby Transformers/TRL stable releases."""

    signature = inspect.signature(callable_object)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _common_args(config: RecipeConfig) -> dict[str, Any]:
    training = config.training
    runtime = resolve_runtime(config)
    values: dict[str, Any] = {
        "output_dir": str(training.output_dir),
        "num_train_epochs": training.num_train_epochs,
        "max_steps": training.max_steps,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "warmup_ratio": training.warmup_ratio,
        "logging_steps": training.logging_steps,
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "gradient_checkpointing": training.gradient_checkpointing,
        "bf16": runtime.bf16,
        "fp16": runtime.fp16,
        "tf32": runtime.tf32,
        "use_cpu": runtime.device == "cpu",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "seed": training.seed,
        "data_seed": training.seed,
        "report_to": training.report_to,
        "deepspeed": training.deepspeed,
        "fsdp": training.fsdp,
        "remove_unused_columns": False,
    }
    if config.data.eval_split:
        values["eval_strategy"] = "steps"
        values["eval_steps"] = training.eval_steps or training.save_steps
    return values


def _runtime_attention(model: Any, config: RecipeConfig) -> Any:
    attention = resolve_runtime(config).attention
    if attention != "auto" and hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(attention)
    return model


def _load_policy(config: RecipeConfig) -> tuple[Any, Any, Any]:
    adapter = get_model_adapter(config.model.name_or_path)
    processor = adapter.load_processor(config)
    model = _runtime_attention(adapter.load_policy_model(config), config)
    if (
        config.data.modality == Modality.IMAGE_TEXT
        and str(config.tuning.strategy) in {"lora", "qlora"}
        and not config.tuning.modules_to_save
    ):
        projections = adapter.describe_components(model)["multimodal_projection"]
        roots = [
            name
            for name in projections
            if name and not any(name.startswith(other + ".") for other in projections if other)
        ]
        config.tuning.modules_to_save = roots
    model = apply_tuning_strategy(model, config)
    return adapter, processor, model


def _build_sft_like(config: DAPTConfig | SFTConfig, dataset: Any, eval_dataset: Any | None) -> Any:
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    _, processor, model = _load_policy(config)
    sample = next(iter(dataset))
    conversational = "messages" in sample or isinstance(sample.get("prompt"), list)
    values = _common_args(config) | {
        "max_length": config.data.max_length,
        "packing": config.data.packing,
        "completion_only_loss": False if config.method == Method.DAPT else None,
        "assistant_only_loss": (
            config.assistant_only_loss and conversational
            if isinstance(config, SFTConfig)
            else False
        ),
        "dataset_num_proc": config.data.num_proc,
    }
    args = TRLSFTConfig(**_supported_kwargs(TRLSFTConfig, values))
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )


def _build_dpo(config: DPOConfig, dataset: Any, eval_dataset: Any | None) -> Any:
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    _, processor, model = _load_policy(config)
    values = _common_args(config) | {
        "max_length": config.data.max_length,
        "beta": config.beta,
        "loss_type": config.loss_type,
    }
    args = TRLDPOConfig(**_supported_kwargs(TRLDPOConfig, values))
    return DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )


def _build_reward(config: RewardConfig, dataset: Any, eval_dataset: Any | None) -> Any:
    from transformers import AutoTokenizer
    from trl import RewardConfig as TRLRewardConfig
    from trl import RewardTrainer

    checkpoint = config.model.reward_name_or_path or config.model.name_or_path
    adapter = get_model_adapter(config.model.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    model = apply_tuning_strategy(
        _runtime_attention(adapter.load_reward_model(config, checkpoint), config), config
    )
    values = _common_args(config) | {"max_length": config.data.max_length}
    args = TRLRewardConfig(**_supported_kwargs(TRLRewardConfig, values))
    return RewardTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def _tokenize_prompts(dataset: Any, processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", processor)

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            text = str(prompt)
        encoded = tokenizer(text, padding=False)
        return {"input_ids": encoded["input_ids"]}

    return dataset.map(tokenize)


def _build_ppo(config: PPOConfig, dataset: Any, eval_dataset: Any | None) -> Any:
    from finetunelab.ppo import EducationalPPOTrainer

    adapter, processor, policy = _load_policy(config)
    reference_name = config.model.reference_name_or_path or config.model.name_or_path
    reward_name = config.model.reward_name_or_path
    value_name = config.model.value_name_or_path or reward_name
    if not reward_name or not value_name:
        raise FineTuneLabError(
            "PPO requires model.reward_name_or_path and model.value_name_or_path "
            "(value may default to reward when reward is provided)."
        )
    reference = _runtime_attention(adapter.load_named_policy_model(config, reference_name), config)
    reward = _runtime_attention(adapter.load_reward_model(config, reward_name), config)
    value = _runtime_attention(adapter.load_reward_model(config, value_name), config)
    return EducationalPPOTrainer(
        config=config,
        tokenizer=getattr(processor, "tokenizer", processor),
        policy=policy,
        reference=reference,
        reward_model=reward,
        value_model=value,
        train_dataset=_tokenize_prompts(dataset, processor),
        eval_dataset=(
            _tokenize_prompts(eval_dataset, processor) if eval_dataset is not None else None
        ),
    )


def _build_grpo(config: GRPOConfig, dataset: Any, eval_dataset: Any | None) -> Any:
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    _, processor, model = _load_policy(config)
    values = _common_args(config) | {
        "max_prompt_length": None if config.data.max_length is None else config.data.max_length,
        "max_completion_length": config.generation.max_new_tokens,
        "num_generations": config.num_generations,
        "beta": config.beta,
        "log_completions": config.logging.log_completions,
    }
    args = TRLGRPOConfig(**_supported_kwargs(TRLGRPOConfig, values))
    return GRPOTrainer(
        model=model,
        reward_funcs=build_reward_functions(config.rewards),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )


def _build_distillation(config: DistillationConfig, dataset: Any, eval_dataset: Any | None) -> Any:
    from trl import DistillationConfig as TRLDistillationConfig
    from trl import DistillationTrainer

    adapter, processor, student = _load_policy(config)
    values = _common_args(config) | {
        "max_completion_length": config.generation.max_new_tokens,
        "beta": config.beta,
        "temperature": config.teacher_temperature,
    }
    args = TRLDistillationConfig(**_supported_kwargs(TRLDistillationConfig, values))
    return DistillationTrainer(
        model=student,
        teacher_model=_runtime_attention(
            adapter.load_named_policy_model(config, config.model.teacher_name_or_path), config
        ),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )


def build_trainer(config: RecipeConfig, dataset: Any, eval_dataset: Any | None = None) -> Any:
    """Build the trainer selected by the discriminated recipe configuration."""

    activate_runtime(resolve_runtime(config))
    if isinstance(config, (DAPTConfig, SFTConfig)):
        return _build_sft_like(config, dataset, eval_dataset)
    if isinstance(config, DPOConfig):
        return _build_dpo(config, dataset, eval_dataset)
    if isinstance(config, RewardConfig):
        return _build_reward(config, dataset, eval_dataset)
    if isinstance(config, PPOConfig):
        return _build_ppo(config, dataset, eval_dataset)
    if isinstance(config, GRPOConfig):
        return _build_grpo(config, dataset, eval_dataset)
    if isinstance(config, DistillationConfig):
        return _build_distillation(config, dataset, eval_dataset)
    raise FineTuneLabError(f"No trainer factory exists for method {config.method}")
