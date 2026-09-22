# Selecting CPU, MPS, or CUDA

FineTuneLab supports explicit backend selection for notebook experiments and CLI
training, evaluation, inspection, feedback generation, and export. Select a backend
before loading a model. A requested backend that is unavailable raises an error;
the application does not silently switch devices.

## CLI and YAML

From the repository root:

```bash
.venv/bin/ftlab doctor
.venv/bin/ftlab train -c configs/qwen35/2b/sft_lora_portable.yaml --device mps
```

Replace `mps` with `cpu`, `cuda`, or `auto`. The portable example uses the local
`../models/Qwen3.5-2B` snapshot, rank-four LoRA, sequence length 128, batch size one,
gradient checkpointing, and two optimizer steps. It is a functional experiment,
not a model-quality benchmark. Change the checkpoint path for other installations.

The equivalent YAML field is:

```yaml
training:
  device: mps
```

Selection precedence is a CLI override, then an explicitly supplied YAML
`training.device`, then `FTLAB_DEVICE`, then `auto`. Explicit `auto` requests
automatic selection instead of consulting a lower-priority `FTLAB_DEVICE` value.
Automatic selection honors `ACCELERATE_USE_CPU=true`; otherwise it chooses CUDA,
MPS, then CPU. An explicit `mps` or `cuda` overrides the Mac kernel's CPU default.
Device changes after Accelerate initializes require a fresh process or kernel;
FineTuneLab never resets an active trainer's shared accelerator state.

`run_manifest.json` records requested/resolved device and effective precision.
`resolved_config.yaml` stores effective settings suitable for replay. `doctor`
reports both CUDA and MPS availability. CUDA launchers retain rank-local GPU
selection for distributed runs.

## Notebooks

The unchanged default `FTLAB_NOTEBOOK_MODE=tiny_cpu` always selects CPU and uses
small random models. To train the downloaded 2B checkpoint on this Mac:

```bash
cd /Users/haroldye/Desktop/Code/FineTuneLab
source .venv/bin/activate
export FTLAB_NOTEBOOK_MODE=local_pretrained
export FTLAB_DEVICE=mps
export FTLAB_LOCAL_MODEL=/Users/haroldye/Desktop/Code/models/Qwen3.5-2B
export FTLAB_USE_QLORA=false
export FTLAB_LORA_RANK=4
export FTLAB_BATCH_SIZE=1
export FTLAB_MAX_LENGTH=128
export FTLAB_MAX_STEPS=2
export FTLAB_ACCUMULATION=1
export FTLAB_NOTEBOOK_OUTPUT=outputs/mps-2b-notebook
jupyter lab notebooks/02_sft.ipynb
```

Select **FineTuneLab (Mac)**, start a fresh kernel, and run all cells. The explicit
MPS selection overrides that kernel's CPU environment setting. The notebook's
first parameter cell prints the resolved settings. Batch size, sequence length,
maximum steps, and accumulation controls above apply to lesson 02; LoRA rank,
device, model path, and dtype apply throughout the course. `FTLAB_MAX_STEPS=0`
runs the entire SFT lesson schedule. The step limit is an upper bound, not a request
to repeat a small dataset beyond its configured epochs.

`FTLAB_DTYPE` can explicitly select `float32`, `bfloat16`, or supported `float16`.
The SFT collator rejects examples with no supervised tokens after truncation.
Image lessons keep their separate visual-token budget guard.

To return to the offline course, unset the `FTLAB_*` settings above and launch with
`FTLAB_NOTEBOOK_MODE=tiny_cpu ACCELERATE_USE_CPU=true`.

## Supported profiles

| Backend | Default weights | Trainer mixed precision | Default attention | LoRA |
|---|---|---|---|---|
| CPU | FP32 | Off | Eager | Unquantized |
| MPS on macOS 14+ | BF16 | Off | Eager | Unquantized |
| CUDA with BF16 support | BF16 | BF16 | SDPA | Existing LoRA/QLoRA support |

Explicit supported weight dtypes are honored. CPU FP16 is rejected. An older CUDA
GPU without BF16 support needs an explicit FP16 or FP32 configuration. Non-CUDA
profiles reject explicitly enabled `training.bf16`, `training.fp16`, or
`training.tf32`; those flags configure Trainer precision and are distinct from
the model weight dtype. When omitted, the defaults above are selected automatically.

FlashAttention 2, DeepSpeed/FSDP, and QLoRA remain CUDA profiles in this project.
Requesting an incompatible combination produces an actionable error; LoRA never
silently replaces a requested QLoRA run. Existing CUDA notebook QLoRA defaults
remain. CPU/MPS notebooks default to ordinary LoRA.

MPS weight loading is sequential to avoid the observed native crash during
concurrent weight conversion. Memory watermarks are not changed. Training does
not automatically enable unsupported-operation CPU fallback. A backend operation
error is reported as a failure; it is not evidence that training completed.

All device paths support selecting full, selective, or LoRA updates subject to
operator support and memory. Passing a short 2B LoRA run does not establish that
4B full training or multi-model PPO/DPO/distillation fits this 16 GB Mac. References,
teachers, and reward/value models use the same selected device; automatic model
offloading is not implemented. Real DAPT still requires a separate Base checkpoint.

## Regression checks

```bash
ACCELERATE_USE_CPU=true .venv/bin/pytest -m 'not gpu' -v
FTLAB_DEVICE=mps ACCELERATE_USE_CPU=false .venv/bin/pytest \
  tests/test_trainer_integration.py tests/test_ppo_integration.py -v
FTLAB_RUN_MPS_TESTS=1 FTLAB_DEVICE=mps ACCELERATE_USE_CPU=false \
  .venv/bin/pytest tests/test_backend_training.py -k mps -v
.venv/bin/ruff check .
.venv/bin/mypy src
```

Run the CPU and MPS commands in separate processes. Ordinary tests never download
production weights. CUDA hardware tests remain opt-in on an NVIDIA host.
