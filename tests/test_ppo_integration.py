from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from finetunelab.config import PPOConfig, load_config
from finetunelab.devices import resolve_runtime
from finetunelab.ppo import EducationalPPOTrainer

ROOT = Path(__file__).parents[1]


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def pad(
        self, rows: list[dict[str, list[int]]], *, padding: bool, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        width = max(len(row["input_ids"]) for row in rows)
        ids = [[0] * (width - len(row["input_ids"])) + row["input_ids"] for row in rows]
        tensor = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": tensor.ne(0).long()}

    def save_pretrained(self, destination: Path) -> None:
        (destination / "tokenizer.txt").write_text("tiny", encoding="utf-8")


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def generate(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        new_token = ((input_ids[:, -1:] + 1) % 7).clamp_min(1)
        return torch.cat([input_ids, new_token, new_token], dim=1)

    def save_pretrained(self, destination: Path, **_: Any) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "model.pt")


class TinyScalar(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 4)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None) -> SimpleNamespace:
        del attention_mask
        pooled = self.embedding(input_ids).mean(dim=1)
        return SimpleNamespace(logits=self.head(pooled))

    def save_pretrained(self, destination: Path, **_: Any) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "model.pt")


@torch.no_grad()
def _clone_policy(source: TinyPolicy) -> TinyPolicy:
    target = TinyPolicy()
    target.load_state_dict(source.state_dict())
    return target


def test_ppo_runs_one_optimization_step(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen35/2b/ppo_lora.yaml")
    assert isinstance(config, PPOConfig)
    config.training.output_dir = tmp_path / "run"
    config.training.max_steps = 1
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_steps = 10
    config.generation.max_new_tokens = 2
    config.num_ppo_epochs = 1
    policy = TinyPolicy()
    trainer = EducationalPPOTrainer(
        config=config,
        tokenizer=TinyTokenizer(),
        policy=policy,
        reference=_clone_policy(policy),
        reward_model=TinyScalar(),
        value_model=TinyScalar(),
        train_dataset=[{"input_ids": [1, 2]}, {"input_ids": [2, 3]}],
        eval_dataset=None,
    )
    expected_device = resolve_runtime(config).device
    assert trainer.accelerator.device.type == expected_device
    for current in (trainer.model, trainer.reference, trainer.reward_model, trainer.value_model):
        assert next(current.parameters()).device.type == expected_device
    result = trainer.train()
    assert result.metrics["steps"] == 1.0
    assert torch.isfinite(torch.tensor(result.metrics["loss/value"]))
    trainer.save_model(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "model.pt").exists()
