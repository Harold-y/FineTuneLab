"""Exercise lesson 00b's masks without loading model weights."""

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).parents[1]
# Minimal Qwen3.5 control flow: only the final assistant turn retains reasoning.
QWEN_TEMPLATE = r"""
{%- set ns = namespace(last_query_index=0) %}
{%- for message in messages %}
    {%- if message.role == 'user' %}{%- set ns.last_query_index = loop.index0 %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = message.content %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = message.reasoning_content|default('') %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n'
                + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- content }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}
"""


@pytest.fixture
def lesson():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["blue green sky grass"],
        trainers.BpeTrainer(
            vocab_size=256,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|im_start|>", "<|im_end|>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        chat_template=QWEN_TEMPLATE,
    )
    notebook = json.loads((ROOT / "notebooks/00b_local_data.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def encode_conversation" in "".join(cell["source"])
    )
    namespace = {"tokenizer": tokenizer, "torch": torch}
    exec(source.split("\nbatch =")[0], namespace)
    return namespace


@pytest.mark.parametrize("turns", [1, 2])
def test_qwen_masks_preserve_text_and_select_each_assistant(lesson, turns):
    tokenizer = lesson["tokenizer"]
    messages = json.loads((ROOT / "examples/education/conversations.jsonl").read_text())["messages"]
    messages = messages[: 2 * turns]
    encoded = lesson["encode_conversation"](messages)
    original = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    assert lesson["render"](messages) == original
    assert encoded["input_ids"] == tokenizer(original, add_special_tokens=False)["input_ids"]
    # Locate targets in the full rendered string; user text and role headers stay ignored.
    spans = []
    for answer in ["blue", "green"][:turns]:
        start = original.index(answer)
        if answer == ["blue", "green"][turns - 1]:
            start = original.index("<think>")
        end = original.index("<|im_end|>", start) + len("<|im_end|>\n")
        spans.append((start, end))
    offsets = tokenizer(original, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    expected = [
        token if any(start <= left < right <= end for start, end in spans) else -100
        for token, (left, right) in zip(encoded["input_ids"], offsets, strict=True)
    ]
    assert encoded["labels"] == expected
    assert encoded["labels"].count(tokenizer.eos_token_id) == turns
    assert tokenizer.chat_template == QWEN_TEMPLATE


def test_padding_can_share_eos_without_masking_real_targets(lesson):
    messages = json.loads((ROOT / "examples/education/conversations.jsonl").read_text())["messages"]
    batch = lesson["collate_text"]([{"messages": messages}, {"messages": messages[:2]}])
    padding = batch["attention_mask"] == 0
    assert padding.any()
    assert (batch["labels"][padding] == -100).all()
    assert (batch["labels"] == lesson["tokenizer"].eos_token_id).sum().item() == 3


def test_template_annotation_is_idempotent_and_preserves_unknown_templates(lesson):
    annotate = lesson["assistant_training_template"]
    annotated = annotate(QWEN_TEMPLATE)
    assert annotated != QWEN_TEMPLATE
    assert annotate(annotated) == annotated
    unknown = "{% for message in messages %}{{ message.content }}{% endfor %}"
    assert annotate(unknown) == unknown
