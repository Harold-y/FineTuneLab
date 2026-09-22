# Dataset Schemas

FineTuneLab loads local JSON, JSONL, and Parquet files or Hugging Face datasets.
Validation samples rows before any model is allocated.

For raw TXT/CSV/JSONL conversion and token-level inspection, run
[lesson 00B](../notebooks/00b_local_data.ipynb). Conversion is explicit; the loader
expects one canonical schema rather than inferring arbitrary column meanings.

## Separate local splits

```yaml
data:
  source: /path/to/canonical-data
  source_type: local
  data_files:
    train: train.jsonl
    validation: validation.jsonl
    test: test.jsonl
  train_split: train
  eval_split: validation
```

Paths in `data_files` are relative to the source directory (or the parent when
`source` is a file). Absolute paths are accepted. All splits use the same builder.
The original single-file `source` form remains supported. For final test evaluation,
set `eval_split: test`; do not select hyperparameters from that result.

## DAPT

```json
{"text": "Unlabeled domain document."}
```

Image-text DAPT may instead use `messages` plus `image` or `images`.

## SFT

```json
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
```

Prompt-completion form is also accepted. For VLM data, add `image` or `images` and
use the content blocks expected by the Qwen processor.

## DPO and reward modeling

```json
{"prompt": "Question", "chosen": "Better answer", "rejected": "Worse answer"}
```

DPO may include images. Reward modeling is text-only in v1.

## PPO, GRPO, and distillation

```json
{"prompt": "Question", "answer": "Reference used by a verifiable reward"}
```

`answer` is optional for PPO/distillation and required by exact/numeric GRPO
rewards. Custom reward functions receive all extra dataset columns.

## Image handling

Relative paths are resolved against `data.image_root`, or against the dataset
file's directory when no root is set. HTTP image URLs and Hugging Face Image/PIL
objects pass through unchanged. VLM configurations default to no truncation.

## Data hygiene

Deduplicate before splitting, keep evaluation prompts out of training, record data
licenses and provenance, and inspect rendered chat templates. Synthetic fixtures in
this repository test mechanics only and must not be used to judge model quality.
