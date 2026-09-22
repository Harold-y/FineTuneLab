"""A compact, readable PPO implementation for text-only RLHF.

TRL 1.13 does not expose its earlier PPO trainer. This implementation keeps the
classic clipped policy/value objective available without pinning the project to an
obsolete TRL release. It favors clarity over rollout-server throughput.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as functional
from accelerate import Accelerator
from torch.utils.data import DataLoader

from finetunelab.config import PPOConfig


class EducationalPPOTrainer:
    """Minimal batched PPO with a frozen reference and reward model."""

    def __init__(
        self,
        *,
        config: PPOConfig,
        tokenizer: Any,
        policy: Any,
        reference: Any,
        reward_model: Any,
        value_model: Any,
        train_dataset: Any,
        eval_dataset: Any | None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = policy
        self.reference = reference
        self.reward_model = reward_model
        self.value_model = value_model
        self.eval_dataset = eval_dataset
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.loader = DataLoader(
            train_dataset,
            batch_size=config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )
        policy_parameters = [item for item in policy.parameters() if item.requires_grad]
        value_parameters = [item for item in value_model.parameters() if item.requires_grad]
        self.policy_optimizer = torch.optim.AdamW(
            policy_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            value_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        prepared = self.accelerator.prepare(
            self.model,
            self.value_model,
            self.policy_optimizer,
            self.value_optimizer,
            self.loader,
        )
        (
            self.model,
            self.value_model,
            self.policy_optimizer,
            self.value_optimizer,
            self.loader,
        ) = prepared
        self.reference.to(self.accelerator.device).eval()
        self.reward_model.to(self.accelerator.device).eval()
        for frozen_model in (self.reference, self.reward_model):
            for parameter in frozen_model.parameters():
                parameter.requires_grad = False

    def _collate(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [{"input_ids": row["input_ids"]} for row in rows]
        return cast(
            dict[str, torch.Tensor],
            self.tokenizer.pad(encoded, padding=True, return_tensors="pt"),
        )

    @staticmethod
    def _completion_logprob(
        model: Any, sequences: torch.Tensor, attention_mask: torch.Tensor, start: int
    ) -> torch.Tensor:
        logits = model(input_ids=sequences, attention_mask=attention_mask).logits[:, :-1]
        targets = sequences[:, 1:]
        token_logprobs = (
            functional.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )
        mask = attention_mask[:, 1:].bool()
        positions = torch.arange(token_logprobs.shape[1], device=sequences.device)
        mask &= positions.unsqueeze(0) >= max(start - 1, 0)
        denominator = mask.sum(dim=1).clamp_min(1)
        return (token_logprobs * mask).sum(dim=1) / denominator

    @staticmethod
    def _scalar(model: Any, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        value = model(input_ids=sequences, attention_mask=attention_mask).logits
        return cast(torch.Tensor, value.reshape(value.shape[0], -1)[:, -1].float())

    def _rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.accelerator.device)
        attention = batch["attention_mask"].to(self.accelerator.device)
        prompt_width = input_ids.shape[1]
        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            sequences = unwrapped.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=self.config.generation.max_new_tokens,
                do_sample=True,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            full_attention = sequences.ne(self.tokenizer.pad_token_id).long()
            old_logprob = self._completion_logprob(
                self.model, sequences, full_attention, prompt_width
            )
            reference_logprob = self._completion_logprob(
                self.reference, sequences, full_attention, prompt_width
            )
            reward = self._scalar(self.reward_model, sequences, full_attention)
            old_value = self._scalar(self.value_model, sequences, full_attention)
            kl = old_logprob - reference_logprob
            target = reward - self.config.kl_coefficient * kl
            advantage = target - old_value
            scale = advantage.std(unbiased=False).clamp_min(1e-6)
            advantage = (advantage - advantage.mean()) / scale
        return {
            "sequences": sequences,
            "attention": full_attention,
            "old_logprob": old_logprob,
            "target": target,
            "advantage": advantage,
            "reward": reward,
            "kl": kl,
            "prompt_width": torch.tensor(prompt_width, device=sequences.device),
        }

    def _update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        clip_range = 0.2
        for _ in range(self.config.num_ppo_epochs):
            with self.accelerator.accumulate(self.model, self.value_model):
                new_logprob = self._completion_logprob(
                    self.model,
                    rollout["sequences"],
                    rollout["attention"],
                    int(rollout["prompt_width"].item()),
                )
                new_value = self._scalar(
                    self.value_model, rollout["sequences"], rollout["attention"]
                )
                ratio = torch.exp(new_logprob - rollout["old_logprob"])
                unclipped = ratio * rollout["advantage"]
                clipped = ratio.clamp(1 - clip_range, 1 + clip_range) * rollout["advantage"]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = functional.mse_loss(new_value, rollout["target"])
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                self.accelerator.backward(policy_loss + 0.5 * value_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                metrics = {
                    "loss/policy": float(policy_loss.detach()),
                    "loss/value": float(value_loss.detach()),
                    "objective/reward": float(rollout["reward"].mean()),
                    "objective/kl": float(rollout["kl"].mean()),
                    "policy/clip_fraction": float(
                        ((ratio - 1.0).abs() > clip_range).float().mean()
                    ),
                }
        return metrics

    def train(self, resume_from_checkpoint: str | None = None) -> SimpleNamespace:
        if resume_from_checkpoint:
            self.accelerator.load_state(resume_from_checkpoint)
        aggregate: defaultdict[str, list[float]] = defaultdict(list)
        step = 0
        maximum = self.config.training.max_steps
        total_epochs = math.ceil(self.config.training.num_train_epochs)
        for _ in range(total_epochs):
            for batch in self.loader:
                metrics = self._update(self._rollout(batch))
                for key, value in metrics.items():
                    aggregate[key].append(value)
                step += 1
                if step % self.config.training.save_steps == 0:
                    checkpoint = Path(self.config.training.output_dir) / f"checkpoint-{step}"
                    self.accelerator.save_state(checkpoint)
                if maximum > 0 and step >= maximum:
                    break
            if maximum > 0 and step >= maximum:
                break
        result = {key: sum(values) / len(values) for key, values in aggregate.items()}
        result["steps"] = float(step)
        return SimpleNamespace(metrics=result)

    def evaluate(self) -> dict[str, float]:
        if self.eval_dataset is None:
            return {}
        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            collate_fn=self._collate,
        )
        rollout = self._rollout(next(iter(loader)))
        return {
            "eval_reward": float(rollout["reward"].mean()),
            "eval_kl": float(rollout["kl"].mean()),
        }

    def save_model(self, output_dir: str) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        policy = self.accelerator.unwrap_model(self.model)
        value = self.accelerator.unwrap_model(self.value_model)
        policy.save_pretrained(destination, safe_serialization=True)
        value.save_pretrained(destination / "value_model", safe_serialization=True)
        self.tokenizer.save_pretrained(destination)
