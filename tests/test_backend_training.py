"""Exercise actual Qwen gradients and adapter reloads on the selected backend."""

import os

import pytest
import torch
from PIL import Image

from finetunelab.config import RECIPE_ADAPTER
from finetunelab.devices import device_rng_state, restore_device_rng_state
from finetunelab.education import make_tiny_checkpoint
from finetunelab.models.qwen35 import Qwen35Adapter
from finetunelab.recipes.factory import _load_policy


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_qwen_adapter_step_and_reload(tmp_path, backend, with_image):
    if backend == "mps" and (
        os.environ.get("FTLAB_RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available()
    ):
        pytest.skip("Opt in with FTLAB_RUN_MPS_TESTS=1 in a fresh MPS process")
    checkpoint = make_tiny_checkpoint(tmp_path / "tiny")
    config = RECIPE_ADAPTER.validate_python(
        {
            "method": "sft",
            "model": {"name_or_path": str(checkpoint), "local_files_only": True},
            "data": {
                "source": str(tmp_path),
                "modality": "image_text" if with_image else "text",
                "max_length": None if with_image else 32,
            },
            "training": {"device": backend, "gradient_checkpointing": True},
            "tuning": {"strategy": "lora", "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0},
        }
    )
    _, processor, model = _load_policy(config)
    assert next(model.parameters()).device.type == backend
    content = [{"type": "text", "text": "What is the color ?"}]
    if with_image:
        content.insert(0, {"type": "image"})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}, {"role": "assistant", "content": "red"}],
        tokenize=False,
    )
    kwargs = {"images": [Image.new("RGB", (32, 32), "red")]} if with_image else {}
    batch = processor(text=[text], return_tensors="pt", **kwargs).to(backend)
    labels = batch["input_ids"].clone()
    labels[labels == model.config.image_token_id] = -100
    before = {name: p.detach().flatten()[:32].cpu().clone() for name, p in model.named_parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()
    loss = model(**batch, labels=labels, use_cache=False).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
    torch.optim.AdamW(trainable, lr=1e-3).step()
    changed = []
    for name, p in model.named_parameters():
        same = torch.equal(before[name], p.detach().flatten()[:32].cpu())
        if not p.requires_grad:
            assert same, name
        elif not same:
            changed.append(name)
    assert changed
    if with_image:
        assert any("merger" in name for name in changed)
    model.eval()
    with torch.no_grad():
        expected = model.generate(**batch, max_new_tokens=2, do_sample=False).cpu()
    artifact = tmp_path / "adapter"
    model.save_pretrained(artifact)
    config.model.adapter_name_or_path = str(artifact)
    reloaded = Qwen35Adapter().load_policy_model(config).eval()
    with torch.no_grad():
        actual = reloaded.generate(**batch, max_new_tokens=2, do_sample=False).cpu()
    assert torch.equal(expected, actual)


def test_mps_rng_restore():
    if os.environ.get("FTLAB_RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available():
        pytest.skip("Opt in with FTLAB_RUN_MPS_TESTS=1")
    torch.mps.manual_seed(42)
    state = device_rng_state("mps")
    expected = torch.rand(8, device="mps").cpu()
    restore_device_rng_state("mps", state)
    assert torch.equal(expected, torch.rand(8, device="mps").cpu())
