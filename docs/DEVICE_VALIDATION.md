# Device selection validation — 2026-09-21

Tested on the M3 MacBook Air with 16 GB unified memory, macOS 26.6.2,
Python 3.11.16, PyTorch 2.14.0, Transformers 5.17.0, and TRL 1.13.0.
The existing pinned Mac dependencies and downloaded checkpoints were retained.

## Real 2B MPS training

Both runs used the local official Qwen3.5-2B checkpoint, BF16 weights, eager
attention, unquantized rank-four LoRA, gradient checkpointing, batch size one,
maximum sequence length 128, and two optimizer updates. MPS memory protections
remained enabled. Neither run used CPU fallback for training operations.

| Check | Notebook 02 | CLI portable recipe |
|---|---|---|
| Two optimizer updates | Passed | Passed |
| Finite losses and gradients | Passed, explicit cell assertions | Passed, finite logged losses/gradient norms |
| Adapter updates | 372 parameter samples changed | Saved LoRA B tensors nonzero and finite |
| Frozen weights | Sampled frozen parameters unchanged | Covered by the notebook and tiny model regression checks |
| Save/reload | Greedy tokens match before saving and after reload | Two independent artifact reloads produce identical tokens |
| Backend report | MPS, BF16 weights, no Trainer AMP | MPS, BF16 weights, no Trainer AMP |

Notebook training losses were approximately 5.06094 and 2.74244. CLI mean training
loss was 3.51476. These runs use different small example datasets; the numbers are
functional validation, not a benchmark or evidence of general model improvement.

Saved adapters:

- `outputs/mps-2b-notebook/02/final`
- `outputs/sft-2b-portable/final`

Detailed evidence:

- `outputs/mac-validation/02_sft_mps_2b.ipynb`: executed real-model notebook.
- `outputs/mps-2b-notebook/02/lesson_report.json`: update, freeze, loss and reload checks.
- `outputs/sft-2b-portable/run_manifest.json`: CLI runtime, versions and metrics.
- `outputs/sft-2b-portable/resolved_config.yaml`: replayable effective configuration.
- `outputs/mac-validation/2b-mps-cli.log`: per-step loss and gradient norms.
- `outputs/mac-validation/2b-mps-cli-reload.json`: adapter integrity and independent reload checks.

## Regression coverage

- The full non-CUDA suite passes **65 tests**, including all ten notebooks in
  fresh offline CPU kernels. Three opt-in MPS tests are skipped in that process
  and nine CUDA tests are deselected; all three MPS checks pass separately.
- Eight tiny trainer integration checks pass on MPS: DAPT, conversational and plain
  SFT, DPO, reward modeling, GRPO, distillation, and PPO. Placement assertions cover
  policies and available reference/teacher/reward/value models.
- Three additional MPS checks pass: real Qwen-class text LoRA gradients/reload,
  image LoRA and visual-merger gradients/reload, and MPS RNG restoration.
- Backend tests cover selection precedence, unavailable hardware, incompatible
  profiles, the Mac kernel's CPU override, initialized Accelerate state, CLI
  overrides, and replaying the resolved configuration.
- Ruff and mypy pass; mypy checks all 24 source files.

JUnit XML and text logs are in `outputs/mac-validation/devices-cpu.*`,
`devices-mps-trainers.*`, and `devices-mps-qwen.*`. The MPS-specific tests run in
fresh processes separately from CPU tests. CUDA tests were not executed because
this machine has no NVIDIA GPU.

## Corrections established by testing

- Explicit backend selection overrides the Mac kernel's CPU preference before
  Accelerate initialization; a conflicting initialized backend requires restart.
- The resolved eager-attention setting reaches custom trainer adapters as well
  as Qwen loaders, avoiding MPS SDPA's unsupported dropout path.
- Notebook assistant-mask detection now checks an actual Jinja `generation` block.
  A template's `add_generation_prompt` variable alone does not provide such masks.
  The official checkpoint's prefix-aligned mask path was exercised by the real run.
- Notebook reload steps release training models, optimizers, and graph references
  before allocating independent checkpoint copies.

The real full-size validation covers 2B text LoRA SFT. Full-size 4B training,
full-parameter training, other full-size objectives, distributed CUDA execution,
and non-CUDA quantization were not validated. Device selection is available
throughout the course, but multi-model workloads can exceed this Mac's memory.

See [device selection](DEVICES.md) for launch commands and supported profiles.
