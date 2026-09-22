from pathlib import Path
from types import SimpleNamespace

import torch

from finetunelab.config import load_config
from finetunelab.tuning import apply_tuning_strategy, parameter_report

ROOT = Path(__file__).parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=6))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(6)])
        self.model.norm = torch.nn.LayerNorm(2)
        self.visual = torch.nn.Linear(2, 2)
        self.lm_head = torch.nn.Linear(2, 2)


def test_selective_default_keeps_vision_frozen() -> None:
    config = load_config(ROOT / "configs/qwen35/4b/dapt_selective.yaml")
    model = apply_tuning_strategy(TinyModel(), config)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any("layers.5" in name for name in trainable)
    assert not any("visual" in name for name in trainable)
    assert parameter_report(model)["trainable"] < parameter_report(model)["total"]


def test_full_strategy_unfreezes_everything() -> None:
    config = load_config(ROOT / "configs/qwen35/4b/dapt_selective.yaml")
    config.tuning.strategy = "full"
    model = TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad = False
    apply_tuning_strategy(model, config)
    assert all(parameter.requires_grad for parameter in model.parameters())
