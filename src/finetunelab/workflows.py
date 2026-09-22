"""High-level train, evaluate, feedback, inspection, and export workflows."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from finetunelab.config import RecipeConfig, TuningStrategy
from finetunelab.data import load_datasets, validate_dataset
from finetunelab.data.fingerprint import dataset_fingerprint
from finetunelab.errors import ConfigurationError, FineTuneLabError
from finetunelab.judges import build_judge
from finetunelab.models import get_model_adapter
from finetunelab.recipes import build_trainer
from finetunelab.runtime import (
    prepare_run_directory,
    seed_everything,
    update_manifest,
    write_manifest,
)
from finetunelab.tuning import apply_tuning_strategy, parameter_report


def inspect_model(config: RecipeConfig) -> dict[str, Any]:
    adapter = get_model_adapter(config.model.name_or_path)
    model = apply_tuning_strategy(adapter.load_policy_model(config), config)
    return {
        "family": adapter.family,
        "checkpoint": config.model.name_or_path,
        "parameters": parameter_report(model),
        "component_counts": {
            key: len(value) for key, value in adapter.describe_components(model).items()
        },
    }


def validate_data_workflow(config: RecipeConfig) -> dict[str, Any]:
    train, evaluation = load_datasets(config)
    return {
        "train": validate_dataset(train, config),
        "evaluation": validate_dataset(evaluation, config) if evaluation is not None else None,
        "fingerprint": dataset_fingerprint(train, config),
    }


def train(config: RecipeConfig, resume: str | None = None) -> dict[str, Any]:
    seed_everything(config.training.seed)
    output = prepare_run_directory(config)
    train_dataset, eval_dataset = load_datasets(config)
    fingerprint = dataset_fingerprint(train_dataset, config)
    trainer = build_trainer(config, train_dataset, eval_dataset)
    report = parameter_report(trainer.model)
    manifest_path = write_manifest(
        output, config, dataset_id=fingerprint, parameters=report, status="training"
    )
    checkpoint = resume or config.training.resume_from_checkpoint
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output / "final"))
        metrics = dict(getattr(result, "metrics", {}))
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        update_manifest(manifest_path, status="completed", metrics=metrics)
        return {"output_dir": str(output), "parameters": report, "metrics": metrics}
    except Exception as exc:
        update_manifest(manifest_path, status="failed", error=type(exc).__name__)
        raise


def evaluate(config: RecipeConfig, checkpoint: str | None = None) -> dict[str, Any]:
    """Evaluate a saved artifact; an explicit checkpoint is required for a baseline."""
    _, eval_dataset = load_datasets(config)
    if eval_dataset is None:
        raise ConfigurationError("Evaluation requires data.eval_split")
    target = checkpoint or str(config.training.output_dir / "final")
    if checkpoint is None and not Path(target).is_dir():
        raise ConfigurationError(
            f"No trained artifact at {target}. Supply --checkpoint for an explicit baseline."
        )
    if config.evaluation.max_samples is not None:
        if hasattr(eval_dataset, "select"):
            eval_dataset = eval_dataset.select(
                range(min(len(eval_dataset), config.evaluation.max_samples))
            )
        else:
            from datasets import Dataset

            eval_dataset = Dataset.from_list(list(eval_dataset.take(config.evaluation.max_samples)))
    runtime = config.model_copy(deep=True)
    runtime.tuning.strategy = TuningStrategy.FULL
    runtime.training.report_to = ["none"]
    runtime.training.gradient_checkpointing = False
    adapter_metadata = Path(target) / "adapter_config.json"
    if adapter_metadata.is_file():
        metadata = json.loads(adapter_metadata.read_text(encoding="utf-8"))
        runtime.model.name_or_path = (
            metadata.get("base_model_name_or_path") or config.model.name_or_path
        )
        runtime.model.adapter_name_or_path = target
    else:
        runtime.model.name_or_path = target
        runtime.model.adapter_name_or_path = None
    if str(config.method) == "reward":
        runtime.model.reward_name_or_path = runtime.model.name_or_path
    if str(config.method) == "ppo":
        runtime.model.reference_name_or_path = (
            config.model.reference_name_or_path or config.model.name_or_path
        )
    output = (
        config.training.output_dir / "evaluations" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    )
    runtime.training.output_dir = output
    trainer = build_trainer(runtime, eval_dataset, eval_dataset)
    metrics = dict(trainer.evaluate())
    if str(config.method) in {"sft", "dapt"} and "eval_loss" in metrics:
        metrics["eval_perplexity"] = math.exp(min(float(metrics["eval_loss"]), 700))
    metrics["checkpoint"] = target
    metrics["evaluated_samples"] = len(eval_dataset)
    output = prepare_run_directory(runtime)
    samples = []
    if str(config.method) != "reward" and config.evaluation.generate_samples:
        processor = get_model_adapter(runtime.model.name_or_path).load_processor(runtime)
        samples = _evaluation_samples(trainer.model, processor, eval_dataset, config)
    (output / "generated_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_manifest(
        output,
        runtime,
        dataset_id=dataset_fingerprint(eval_dataset, config),
        parameters=parameter_report(trainer.model),
        status="evaluated",
    )
    return metrics


def _evaluation_samples(
    model: Any, processor: Any, dataset: Any, config: RecipeConfig
) -> list[dict[str, Any]]:
    """Generate from held-out prompts, never from their reference assistant answer."""
    tokenizer = getattr(processor, "tokenizer", processor)
    model.eval()
    samples = []
    for index, row in enumerate(dataset):
        if index >= config.evaluation.generate_samples:
            break
        prompt = row.get("prompt", row.get("messages", row.get("text", "")))
        if isinstance(prompt, list):
            prompt = list(prompt)
            while prompt and prompt[-1]["role"] == "assistant":
                prompt.pop()
            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            text = str(prompt)
        images = row.get("images", row.get("image"))
        kwargs = {"images": images} if images is not None else {}
        inputs = processor(text=[text], return_tensors="pt", **kwargs)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=config.generation.max_new_tokens
            )
        completion = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        samples.append({"prompt": text, "completion": completion})
    return samples


def _prompt_text(prompt: Any, tokenizer: Any) -> str:
    if isinstance(prompt, list):
        return cast(
            str,
            tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
        )
    return str(prompt)


def generate_feedback(config: RecipeConfig) -> dict[str, Any]:
    if config.judge is None:
        raise ConfigurationError("feedback generation requires a judge section")
    if config.data.modality != "text":
        raise ConfigurationError("v1 feedback generation supports text prompts only")
    train_dataset, _ = load_datasets(config)
    adapter = get_model_adapter(config.model.name_or_path)
    processor = adapter.load_processor(config)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = adapter.load_policy_model(config)
    model.eval()
    judge = build_judge(config.judge)
    output = prepare_run_directory(config) / "feedback.jsonl"
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in train_dataset:
            prompt = _prompt_text(row["prompt"], tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                    top_k=config.generation.top_k,
                    max_new_tokens=config.generation.max_new_tokens,
                    num_return_sequences=config.generation.num_candidates,
                )
            prefix = inputs["input_ids"].shape[-1]
            candidates = tokenizer.batch_decode(generated[:, prefix:], skip_special_tokens=True)
            decision = judge.compare(prompt, candidates)
            record = {
                "prompt": row["prompt"],
                "candidates": candidates,
                "chosen": candidates[decision["winner"]],
                "rejected": candidates[
                    min(range(len(candidates)), key=decision["scores"].__getitem__)
                ],
                "judge": {
                    "backend": config.judge.backend,
                    "model": config.judge.model,
                    **decision,
                },
                "generator": {
                    "model": config.model.name_or_path,
                    "revision": config.model.revision,
                    "parameters": config.generation.model_dump(),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return {"path": str(output), "records": count}


def export_model(config: RecipeConfig) -> dict[str, str]:
    """Merge a saved PEFT adapter into its base model and save standard HF artifacts."""

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise FineTuneLabError("Export requires PEFT") from exc
    adapter_path = config.training.output_dir.expanduser().resolve() / "final"
    if not adapter_path.exists():
        raise FineTuneLabError(f"No final checkpoint found at {adapter_path}")
    adapter = get_model_adapter(config.model.name_or_path)
    runtime = config.model_copy(deep=True)
    runtime.tuning.strategy = TuningStrategy.FULL
    runtime.model.adapter_name_or_path = None
    base = adapter.load_policy_model(runtime)
    if not (adapter_path / "adapter_config.json").exists():
        raise FineTuneLabError("The final checkpoint is not a PEFT adapter; no merge is needed")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    destination = config.training.output_dir.expanduser().resolve() / "merged"
    merged.save_pretrained(destination, safe_serialization=True)
    adapter.load_processor(config).save_pretrained(destination)
    return {"path": str(destination)}
