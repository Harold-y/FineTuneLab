# Mac setup validation — 2026-09-21

This records the initial installation. The subsequent [device-selection update](DEVICES.md)
adds MPS training; see `DEVICE_VALIDATION.md` for those results.

Machine: Apple M3 MacBook Air, 16 GB unified memory, ARM64 macOS 26.6.2.
Environment: project `.venv`, managed Python 3.11.16, uv 0.12.17.
Jupyter kernel: **FineTuneLab (Mac)** (`finetunelab-mac`).

## Results

| Check | Result |
|---|---|
| Dependency consistency | 154 installed packages compatible, including editable FineTuneLab |
| Dependency provenance | All 153 pins in `requirements-mac.txt` match `uv.lock` |
| Ruff | Passed |
| mypy | Passed, all 23 source files |
| Non-CUDA pytest suite | **51 passed, 9 CUDA tests deselected**, 41.94 seconds |
| Notebooks | All ten passed in isolated offline CPU kernels |
| CLI | `ftlab doctor` passed |
| Apple GPU | PyTorch MPS built and available |
| Qwen3.5-2B | All 13 files size/hash verified; text and image generation passed on MPS |
| Qwen3.5-4B | All 14 files size/hash verified; text and image generation passed on CPU and MPS |

Both models used BF16, short inputs, four generated tokens, and local files only.
The text prompt was “Name one color.” The image check added a 64×64 red image.
2B generated `Blue` and `Red`; 4B generated `Blue` and `red`. These are functional
smoke checks, not model-quality benchmarks or full training tests.

## Compatibility corrections

- The initial test run mixed MPS model weights with CPU batches. FineTuneLab now
  forwards `ACCELERATE_USE_CPU=true` explicitly to Transformers' `use_cpu` option.
  The registered Mac kernel sets this environment variable. The corrected complete
  suite passed without changing or skipping any notebook cells.
- Initial asynchronous 4B loading into MPS exited with native SIGSEGV (139).
  The native stack implicated MPS shader initialization during tensor copying.
  The smoke checker now disables asynchronous loading. The sequential MPS retry
  passed both cases; an independent CPU run also passed. MPS memory protections
  were not relaxed.
- Qwen's optional accelerated convolution/DeltaNet kernels are absent. Transformers
  used its reference PyTorch implementations successfully.

## Installed core versions

| Package | Version |
|---|---|
| torch / torchvision | 2.14.0 / 0.29.0 |
| transformers / trl | 5.17.0 / 1.13.0 |
| peft / accelerate | 0.21.0 / 1.15.0 |
| datasets / numpy | 4.8.5 / 2.2.6 |
| huggingface-hub | 1.32.0 |
| jupyterlab / ipykernel | 4.6.3 / 7.3.0 |

All dependency versions are listed in `../requirements-mac.txt`.

## Evidence and usage

Local validation artifacts are under `../outputs/mac-validation/`:

- `environment.json`: installed package versions, hardware backend checks, kernel configuration.
- `pytest.xml` and `pytest.log`: full test results, including all notebook cases.
- `2b-mps.json`, `4b-mps.json`, `4b-cpu.json` and corresponding logs: model checks.
- `4b-mps-initial.json` and `.log`: recorded initial native failure.

Snapshots are in `/Users/haroldye/Desktop/Code/models/Qwen3.5-2B` and
`/Users/haroldye/Desktop/Code/models/Qwen3.5-4B`. The adjacent
`Qwen3.5-2B.verification.json` and `Qwen3.5-4B.verification.json` contain pinned
revisions and every verified file hash. Total model snapshot size is
13,914,181,492 bytes, excluding downloader metadata.

See [Mac setup](MAC_SETUP.md) for reproducible installation and launch commands.
The full course is validated in `tiny_cpu` mode. Existing `local_pretrained`
training notebooks required CUDA at this initial setup stage; full-size fine-tuning and simultaneous
teacher/student loading were not part of this validation.
