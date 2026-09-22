"""Typed experiment configuration and cross-field capability validation."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from finetunelab.errors import ConfigurationError


class Method(StrEnum):
    """Training recipes exposed by the framework."""

    DAPT = "dapt"
    SFT = "sft"
    DPO = "dpo"
    REWARD = "reward"
    PPO = "ppo"
    GRPO = "grpo"
    DISTILLATION = "distillation"


class TuningStrategy(StrEnum):
    """Ways in which model parameters may be updated."""

    FULL = "full"
    SELECTIVE = "selective"
    LORA = "lora"
    QLORA = "qlora"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE_TEXT = "image_text"


SUPPORTED_QWEN35_MODELS = frozenset(
    {
        "Qwen/Qwen3.5-2B-Base",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-4B",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictModel):
    name_or_path: str
    local_files_only: bool = False
    base_checkpoint: str | None = None
    adapter_name_or_path: str | None = None
    revision: str = "main"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "bfloat16"
    attention_implementation: Literal["auto", "eager", "sdpa", "flash_attention_2"] = "sdpa"
    trust_remote_code: bool = False
    teacher_name_or_path: str | None = None
    reference_name_or_path: str | None = None
    reward_name_or_path: str | None = None
    value_name_or_path: str | None = None


class DataConfig(StrictModel):
    source: str
    source_type: Literal["auto", "local", "hub"] = "auto"
    train_split: str = "train"
    eval_split: str | None = None
    data_files: dict[str, str] | None = None
    format: Literal["auto", "json", "jsonl", "parquet"] = "auto"
    modality: Modality = Modality.TEXT
    image_root: Path | None = None
    streaming: bool = False
    packing: bool = False
    max_length: int | None = 2048
    num_proc: int | None = None

    @model_validator(mode="after")
    def validate_multimodal_length(self) -> DataConfig:
        if self.modality == Modality.IMAGE_TEXT and self.max_length is not None:
            raise ValueError(
                "Image-text recipes default to max_length=null so visual tokens cannot be "
                "silently truncated. Set it only after validating every sample."
            )
        return self


class TuningConfig(StrictModel):
    strategy: TuningStrategy = TuningStrategy.LORA
    selective_preset: Literal["last_n_text_layers", "norms_and_head", "custom"] = (
        "last_n_text_layers"
    )
    last_n_layers: int = Field(default=4, ge=1)
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=lambda: [r"(?:visual|vision)"])
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_target_modules: list[str] | Literal["all-linear"] = "all-linear"
    modules_to_save: list[str] = Field(default_factory=list)
    quant_compute_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True


class TrainingConfig(StrictModel):
    output_dir: Path = Path("outputs/run")
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = -1
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int | None = Field(default=None, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    seed: int = 42
    report_to: list[str] = Field(default_factory=lambda: ["none"])
    resume_from_checkpoint: str | None = None
    deepspeed: str | None = None
    fsdp: str | None = None

    @model_validator(mode="after")
    def validate_precision(self) -> TrainingConfig:
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive")
        return self


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, gt=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    num_candidates: int = Field(default=2, ge=2)


class EvaluationConfig(StrictModel):
    enabled: bool = True
    max_samples: int | None = Field(default=128, ge=1)
    generate_samples: int = Field(default=4, ge=0)


class LoggingConfig(StrictModel):
    log_completions: bool = False
    redact_keys: list[str] = Field(default_factory=lambda: ["token", "api_key", "secret"])


class RewardSpec(StrictModel):
    name: Literal["exact_match", "numeric", "regex", "format", "length", "python"]
    weight: float = 1.0
    pattern: str | None = None
    callable: str | None = None
    minimum: float = 0.0
    maximum: float = 1.0


class JudgeConfig(StrictModel):
    backend: Literal["local", "openai_compatible"] = "local"
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    prompt_template: str = (
        "Choose the better response for the user request. Return JSON with keys "
        "winner (0 or 1), scores (two numbers), and reason."
    )


class CommonRecipeConfig(StrictModel):
    schema_version: Literal[1] = 1
    method: Method
    model: ModelConfig
    data: DataConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    judge: JudgeConfig | None = None
    run_name: str | None = None

    @model_validator(mode="after")
    def validate_capability_matrix(self) -> CommonRecipeConfig:
        model_path = Path(self.model.name_or_path)
        is_local = model_path.exists() or self.model.name_or_path.startswith((".", "/"))
        if not is_local and self.model.name_or_path not in SUPPORTED_QWEN35_MODELS:
            raise ValueError(
                f"Unsupported initial checkpoint: {self.model.name_or_path}. "
                "FineTuneLab v1 supports the four registered Qwen3.5 checkpoints "
                "or a local derivative."
            )
        if self.method in {Method.REWARD, Method.PPO} and self.data.modality != Modality.TEXT:
            raise ValueError("Reward modeling and PPO are text-only in FineTuneLab v1")
        if (
            self.method in {Method.REWARD, Method.PPO}
            and self.tuning.strategy == TuningStrategy.QLORA
        ):
            raise ValueError("QLoRA is not supported for reward modeling or PPO in v1")
        if self.method == Method.DAPT:
            identity = self.model.base_checkpoint or self.model.name_or_path
            if model_path.is_dir() and self.model.base_checkpoint is None:
                metadata = model_path / "config.json"
                if metadata.exists():
                    saved = json.loads(metadata.read_text(encoding="utf-8"))
                    identity = saved.get(
                        "finetunelab_base_checkpoint", saved.get("_name_or_path", "")
                    )
            if not identity.endswith("-Base"):
                raise ValueError(
                    "DAPT requires a Base checkpoint identity. For a renamed local snapshot, "
                    "set model.base_checkpoint to its original Qwen/*-Base repository ID."
                )
        return self


class DAPTConfig(CommonRecipeConfig):
    method: Literal[Method.DAPT]


class SFTConfig(CommonRecipeConfig):
    method: Literal[Method.SFT]
    assistant_only_loss: bool = True


class DPOConfig(CommonRecipeConfig):
    method: Literal[Method.DPO]
    beta: float = Field(default=0.1, gt=0)
    loss_type: Literal["sigmoid", "hinge", "ipo", "robust"] = "sigmoid"


class RewardConfig(CommonRecipeConfig):
    method: Literal[Method.REWARD]
    margin: float | None = None


class PPOConfig(CommonRecipeConfig):
    method: Literal[Method.PPO]
    kl_coefficient: float = Field(default=0.05, ge=0)
    num_ppo_epochs: int = Field(default=4, ge=1)


class GRPOConfig(CommonRecipeConfig):
    method: Literal[Method.GRPO]
    rewards: list[RewardSpec] = Field(default_factory=lambda: [RewardSpec(name="exact_match")])
    num_generations: int = Field(default=4, ge=2)
    beta: float = Field(default=0.04, ge=0)


class DistillationConfig(CommonRecipeConfig):
    method: Literal[Method.DISTILLATION]
    beta: float = Field(default=1.0, ge=0, le=1)
    teacher_temperature: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def require_teacher(self) -> DistillationConfig:
        if not self.model.teacher_name_or_path:
            raise ValueError("distillation requires model.teacher_name_or_path")
        return self


RecipeConfig: TypeAlias = Annotated[
    DAPTConfig | SFTConfig | DPOConfig | RewardConfig | PPOConfig | GRPOConfig | DistillationConfig,
    Field(discriminator="method"),
]
RECIPE_ADAPTER: TypeAdapter[RecipeConfig] = TypeAdapter(RecipeConfig)


def load_config(path: str | Path) -> RecipeConfig:
    """Load and validate a YAML or JSON experiment configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        if config_path.suffix.lower() == ".json":
            raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cast(RecipeConfig, RECIPE_ADAPTER.validate_python(raw))
    except (ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc


def dump_config(config: RecipeConfig, path: str | Path) -> None:
    """Write the fully resolved configuration without serializing Python objects."""

    data = config.model_dump(mode="json")
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def redacted_dict(config: RecipeConfig) -> dict[str, Any]:
    """Return a recursively redacted representation safe for logs and manifests."""

    blocked = {item.lower() for item in config.logging.redact_keys}

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "***REDACTED***"
                if any(word in key.lower() for word in blocked)
                else visit(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return cast(dict[str, Any], visit(config.model_dump(mode="json")))
