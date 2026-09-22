"""Offline fixtures and display utilities; lesson objectives live in the notebooks.

The tiny checkpoint uses the official Qwen3.5 classes with reduced dimensions.
Its tokenizer and random weights are synthetic, not pretrained Qwen assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def project_root(start: Path | None = None) -> Path:
    """Find the repository from either its root or the notebooks directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "finetunelab").is_dir():
            return candidate
    raise FileNotFoundError("Start Jupyter from FineTuneLab or its notebooks directory")


def tiny_config(vocab_size: int, token_ids: dict[str, int]) -> Any:
    """Keep one complete three-linear/one-full attention cycle and a tiny ViT."""
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    text = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
        pad_token_id=token_ids["<pad>"],
        eos_token_id=token_ids["<eos>"],
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        patch_size=8,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    return Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=token_ids["<|image_pad|>"],
        video_token_id=token_ids["<|video_pad|>"],
        vision_start_token_id=token_ids["<|vision_start|>"],
        vision_end_token_id=token_ids["<|vision_end|>"],
    )


def make_tiny_checkpoint(destination: Path, seed: int = 42) -> Path:
    """Create an isolated, reproducible local checkpoint without network access."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessorPil,
        Qwen3_5ForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    destination.mkdir(parents=True, exist_ok=True)
    special = [
        "<pad>",
        "<eos>",
        "<unk>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<user>",
        "<assistant>",
        "<system>",
    ]
    words = [
        "What",
        "is",
        "the",
        "color",
        "of",
        "sky",
        "grass",
        "image",
        "?",
        "blue",
        "green",
        "red",
        ".",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "+",
        "=",
        "Answer",
        "briefly",
        "good",
        "bad",
        "correct",
        "wrong",
        "yes",
        "no",
        "domain",
        "document",
        "Water",
        "freezes",
        "at",
        "zero",
        "degrees",
        "Ice",
        "cold",
        "hot",
        "Sun",
        "bright",
        "A",
        "B",
        "C",
        "D",
        "explain",
        "count",
        "square",
        "circle",
        "number",
        "answer",
        "user",
        "assistant",
    ]
    vocabulary = {word: index for index, word in enumerate(dict.fromkeys(special + words))}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=special[3:],
        model_max_length=512,
        model_input_names=["input_ids", "attention_mask"],
    )
    # Generation tags expose the exact assistant span to Transformers/TRL.
    # Jinja blocks cannot be opened conditionally: use a macro for common content.
    template = (
        "{% macro body(message) %}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
        "{% endfor %}{% endif %}{{ eos_token }}{% endmacro %}"
        "{% for message in messages %}{{ '<' + message['role'] + '>' }}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ body(message) }}{% endgeneration %}"
        "{% else %}{{ body(message) }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    tokenizer.chat_template = template
    image_processor = Qwen2VLImageProcessorPil(
        patch_size=8,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 32 * 32, "longest_edge": 32 * 32},
    )
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=template,
        video_processor=Qwen3VLVideoProcessor(),
    )
    processor.save_pretrained(destination)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        config = tiny_config(len(tokenizer), vocabulary)
        config._attn_implementation = "eager"
        config.finetunelab_base_checkpoint = "synthetic-tiny-Base"
        config.finetunelab_fixture = True
        model = Qwen3_5ForConditionalGeneration(config)
        model.save_pretrained(destination)
    return destination


def inspect_local_checkpoint(path: Path) -> dict[str, Any]:
    """Fail early for an incomplete snapshot, without resolving a Hub ID."""
    if not path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {missing} in {path}")
    if not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"No local model weights found in {path}")
    for index in path.glob("*.index.json"):
        metadata = json.loads(index.read_text(encoding="utf-8"))
        absent = {
            shard
            for shard in metadata.get("weight_map", {}).values()
            if not (path / shard).is_file()
        }
        if absent:
            raise FileNotFoundError(f"Missing weight shards: {sorted(absent)}")
    return {
        "directory": str(path.resolve()),
        "config": json.loads((path / "config.json").read_text(encoding="utf-8")),
        "files": sorted(item.name for item in path.iterdir() if item.is_file()),
    }


def token_table(tokenizer: Any, batch: dict[str, Any], row: int = 0) -> list[dict[str, Any]]:
    """Make visible which positions can attend and which contribute to loss."""
    ids = batch["input_ids"][row].tolist()
    labels = batch.get("labels", batch["input_ids"])[row].tolist()
    attention = batch["attention_mask"][row].tolist()
    return [
        {
            "position": i,
            "id": token,
            "token": tokenizer.convert_ids_to_tokens(token),
            "attention": attention[i],
            "label": labels[i],
            "supervised": labels[i] != -100,
        }
        for i, token in enumerate(ids)
    ]
