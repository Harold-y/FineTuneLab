# Learn FineTuneLab from tensors to artifacts

Start with **00A → 00B → 02 → 08**, then study the other objectives. Every notebook runs independently in a fresh kernel. Code that computes masks, losses, rollouts, gradients and optimizer updates is visible in notebook cells. `finetunelab.education` contains only fixture/setup/inspection utilities.

## New to fine-tuning?

Basic Python is enough to start reading. Before the first code cell, each chapter
now introduces its terms in plain language, explains the problem and its limits,
shows an example, walks through the objective and its symbols, connects them to
code variables, and offers a quick check with expandable answers. You do not need
to memorize acronyms or read another document first.

The shared **fictional NovaBot manual** describes three modes: `idle`, `mapping`,
and `navigation`. Watch that same fact become raw text, a demonstration answer,
a preference pair, a verified reward, or a teacher distribution. The examples and
hand-calculated numbers explain training signals; they are not measured results.
The existing runnable synthetic datasets remain unchanged, and no example promises
that a randomly initialized model will learn the stated answer in one update.

Keep three questions separate: **what objective is optimized**, **which parameters
are updated**, and **who or what supplies feedback**. For example, Supervised
Fine-Tuning (SFT) can use Low-Rank Adaptation (LoRA) on human demonstrations. Neither
the objective nor the feedback source determines the parameter strategy.

Use the [English glossary](../docs/GLOSSARY.md) for review and the
[training-method comparison](../docs/TRAINING_METHODS.md) to choose an objective.
All chapters keep their own explanations so they can also be read independently.

## Course map

| Lesson | Prerequisites | Main experiment |
|---|---|---|
| [00A: Qwen in PyTorch](00a_qwen_in_pytorch.ipynb) | Basic Python | Understand models/weights and inspect official modules, local files and forward hooks |
| [00B: Local data](00b_local_data.ipynb) | Python file I/O | TXT/CSV/JSONL conversion, grouped splits, masks and image paths |
| [01: Domain-Adaptive Pre-Training](01_dapt.ipynb) | 00A, 00B | Raw text as supervision, document boundaries, packing and causal training |
| [02: Supervised Fine-Tuning](02_sft.ipynb) | 00A, 00B | Demonstrations, teacher forcing, assistant-only labels and a PyTorch loop |
| [03: Direct Preference Optimization](03_dpo.ipynb) | 02; log-probabilities explained inline | Chosen/rejected pairs and preference margins against a frozen reference |
| [04: Reinforcement Learning from Human / AI Feedback](04_rlhf_rlaif.ipynb) | 02 | Reward models, rollouts, advantage and educational sequence-level Proximal Policy Optimization |
| [05: Group Relative Policy Optimization / Reinforcement Learning with Verifiable Rewards](05_grpo_rlvr.ipynb) | 02 | Verifiers, group comparisons, reward variance and clipped updates |
| [06: Generalized Knowledge Distillation](06_gkd.ipynb) | 02; divergence explained inline | Teacher/student roles, soft targets, temperature and on-policy trajectories |
| [07: Image SFT](07_image_sft.ipynb) | SFT | Image processor, patches, visual masks and merger updates |
| [08: Evaluation/artifacts](08_evaluation_artifacts.ipynb) | SFT | Held-out evaluation, exact resume, reload, merge and CLI evaluation |

Each chapter has learning objectives, executable steps, intermediate inspections, interpretation, failure diagnostics and exercises. In advanced lessons, a separate one-step framework experiment connects the explicit math to the Trainer boundary.

## Install and open

On Apple Silicon, follow [Mac setup](../docs/MAC_SETUP.md) for the pinned Mac
dependencies, **FineTuneLab (Mac)** kernel, and local model checks.

From the project root:

```bash
uv sync --locked --extra dev --extra notebooks
uv run jupyter lab notebooks
```

Select the kernel using the project's `.venv` interpreter. Inside a notebook, `import sys; print(sys.executable)` should identify that environment. Do not install a different Transformers version into the kernel after resolving `uv.lock`.

To execute the complete offline course with isolated kernels:

```bash
uv run pytest tests/test_notebooks.py -v
```

The test runner sets offline/cache environment variables, writes an executed notebook into each test's temporary directory and checks experiment reports. Ordinary tests download no model weights.

## Two execution modes

**tiny_cpu (default).** Real official Qwen3.5 model classes, four language layers with a 3:1 linear/full-attention cycle, a one-layer vision encoder, a synthetic tokenizer and randomly initialized weights. No Hub downloads, API keys or CUDA required. The tests check mechanics, not semantic quality. The slower reference PyTorch Gated DeltaNet kernels are sufficient at this scale.

**local_pretrained.** Set the parameter cell or environment variables before launching Jupyter:

```powershell
$env:FTLAB_NOTEBOOK_MODE = "local_pretrained"
$env:FTLAB_LOCAL_MODEL = "D:\Models\Qwen3.5-2B"
$env:FTLAB_LOCAL_TEACHER = "D:\Models\Qwen3.5-4B"
$env:FTLAB_QA_FILE = "D:\Data\qa.csv"
uv run jupyter lab notebooks
```

Linux and macOS use the same variable names with shell `export`. Set `FTLAB_DEVICE`
to `cpu`, `mps`, `cuda`, or `auto` for local checkpoints. CPU/MPS default to
unquantized LoRA; CUDA QLoRA needs the quant extra. See [device selection](../docs/DEVICES.md)
for the tested 2B MPS SFT setup and bounded training controls. Paths use `pathlib`;
use raw Python strings for manually entered Windows paths.

The snapshot must include config, all weight shards, tokenizer assets, processor configuration and chat template. Every loader uses `local_files_only=True`. An incomplete download raises an error instead of silently fetching files. Use the matching Base snapshot for DAPT. Source code inspection does not imply that a default config has the same dimensions as a downloaded 2B/4B model.

CUDA real-model lessons use QLoRA by default where supported; CPU/MPS use LoRA. Reward/PPO uses unquantized policy/reward/reference/value models, and GKD also holds an unquantized teacher: these profiles need larger GPUs. CPU success is not a claim that all real profiles fit in 16 GB. The framework comparison cells in advanced lessons run automatically only in tiny mode; run them in a fresh process for real models.

## Bring your own data

For a renamed local DAPT snapshot whose config no longer records its Hub identity,
set `FTLAB_BASE_CHECKPOINT` to `Qwen/Qwen3.5-2B-Base` or `Qwen/Qwen3.5-4B-Base`,
matching the checkpoint you downloaded. The equivalent framework setting is
`model.base_checkpoint`. It records provenance; it does not convert post-trained
weights into Base weights.

- `FTLAB_QA_FILE`: CSV or JSONL with `group_id`, `question`, `answer`; preference lessons also require a distinct nonempty `rejected`.
- `FTLAB_LESSON_DATA`: directory containing the introductory `domain.txt`, `instructions.jsonl` and `conversations.jsonl` examples. Copy and adapt the supplied fixture directory.
- `FTLAB_IMAGE_DATA`: image SFT JSONL with `image` and `messages`, optionally `group_id`.
- `FTLAB_IMAGE_ROOT`: root for relative image references; defaults to the annotation directory.
- `FTLAB_NOTEBOOK_OUTPUT`: experiment output root; defaults to `outputs/notebooks`.

Conversion is explicit. Rename your columns in the conversion cell instead of relying on guessed meanings. Split by document/conversation/image group before chunking; remove duplicates before splitting. Tiny one-record held-out sets illustrate the pipeline only.

## How to read the results

The train/validation/test sets have different roles. Loss must be finite, but it need not improve after a single synthetic update. Reconstructed masks, correct frozen/trainable updates and identical reload tokens are the primary mechanics checks.

The RLHF chapter labels its PPO as a sequence-level teaching approximation; it does not implement per-token GAE. The GRPO chapter uses a length reward on random outputs only to make the signal visible; correctness requires task-specific verifiers. The GKD chapter checks its divergence against the locked TRL kernel.

Each training chapter saves `final/`, a processor and `lesson_report.json`. The artifacts chapter separately saves optimizer/RNG/position state, verifies resumed parameters and AdamW state, merges an unquantized base, and evaluates the saved checkpoint. Keep the base snapshot with every adapter.

## Project maps

Read [execution and data flow](../docs/EXECUTION_AND_DATA_FLOW.md), [model architecture](../docs/QWEN35_ARCHITECTURE.md) and [module responsibilities](../docs/PROJECT_ARCHITECTURE.md). Trace a notebook operation to the corresponding production module only after understanding its intermediate tensors.
