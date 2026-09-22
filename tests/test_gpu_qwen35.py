import os
from pathlib import Path

import pytest
import torch

from finetunelab.config import load_config
from finetunelab.workflows import train

ROOT = Path(__file__).parents[1]
RUN_GPU = os.getenv("FTLAB_RUN_GPU_TESTS") == "1"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "relative",
    [
        "configs/qwen35/2b/dapt_lora.yaml",
        "configs/qwen35/2b/sft_qlora.yaml",
        "configs/qwen35/2b/dpo_lora.yaml",
        "configs/qwen35/2b/grpo_qlora.yaml",
        "configs/qwen35/2b/distillation_qlora.yaml",
    ],
)
def test_qwen35_qlora_one_step(relative: str, tmp_path: Path) -> None:
    if not RUN_GPU or not torch.cuda.is_available():
        pytest.skip("Set FTLAB_RUN_GPU_TESTS=1 on a CUDA host")
    config = load_config(ROOT / relative)
    config.tuning.strategy = "qlora"
    config.training.output_dir = tmp_path / Path(relative).stem
    config.training.max_steps = 1
    config.training.save_steps = 10
    result = train(config)
    assert result["metrics"]


@pytest.mark.gpu
@pytest.mark.parametrize("size", ["2B", "4B"])
@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_local_qwen_snapshot_generate(size: str, with_image: bool) -> None:
    """Opt-in smoke check; never download a production checkpoint implicitly."""
    if not RUN_GPU or not torch.cuda.is_available():
        pytest.skip("Set FTLAB_RUN_GPU_TESTS=1 on a CUDA host")
    checkpoint = os.getenv(f"FTLAB_QWEN35_{size}")
    if not checkpoint:
        pytest.skip(f"Set FTLAB_QWEN35_{size} to an already downloaded local snapshot")
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from finetunelab.education import inspect_local_checkpoint

    inspect_local_checkpoint(Path(checkpoint))
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    content = [{"type": "text", "text": "Name one color."}]
    if with_image:
        content.insert(0, {"type": "image"})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    kwargs = {"images": [Image.new("RGB", (64, 64), "red")]} if with_image else {}
    batch = processor(text=[text], return_tensors="pt", **kwargs).to("cuda")
    with torch.no_grad():
        generated = model.generate(**batch, max_new_tokens=2, do_sample=False)
    assert generated.shape[-1] > batch["input_ids"].shape[-1]
