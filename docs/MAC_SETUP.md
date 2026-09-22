# Apple Silicon Mac setup

This environment targets ARM64 macOS 14 or newer with Python 3.11. The initial
machine is an M3 MacBook Air with 16 GB unified memory and macOS 26.6.2.
`requirements-mac.txt` pins the runtime, notebook, development, and Hugging Face
dependencies from `uv.lock`, resolved for this Mac. The original project dependency
definitions and lockfile remain the cross-platform source of truth.

## Install or reproduce

Run from the FineTuneLab project root:

```bash
export PATH="$HOME/.local/bin:$PATH"
# If uv is absent:
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
uv python install 3.11
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements-mac.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m ipykernel install --user --name finetunelab-mac \
  --display-name 'FineTuneLab (Mac)' --env ACCELERATE_USE_CPU true
```

The Mac environment includes PyTorch's CPU and MPS backends. Optional quantization,
distributed training, and experiment tracking extras are not installed for this
course/testing setup. No CUDA-specific packages are required. System Python and
shell startup files are not modified.

To regenerate the Mac dependency file from the lockfile, on this Mac:

```bash
uv export --locked --extra dev --extra notebooks --no-emit-project --no-hashes \
  --output-file /tmp/finetunelab-lock-export.txt > /dev/null
uv pip compile /tmp/finetunelab-lock-export.txt --python .venv/bin/python \
  --only-binary :all: --no-annotate \
  --custom-compile-command 'See docs/MAC_SETUP.md for regeneration from uv.lock on Apple Silicon' \
  -o requirements-mac.txt
```

This uses the actual macOS version; the generic `aarch64-apple-darwin` target can
default to macOS 13 and reject the macOS 14+ PyTorch wheels.

## Open the course

```bash
cd /Users/haroldye/Desktop/Code/FineTuneLab
source .venv/bin/activate
FTLAB_NOTEBOOK_MODE=tiny_cpu ACCELERATE_USE_CPU=true jupyter lab notebooks
```

Select **FineTuneLab (Mac)**. Start with 00A → 00B → 02 → 08. `import sys;
print(sys.executable)` should point to this project's `.venv/bin/python`.
The kernel sets `ACCELERATE_USE_CPU=true` so the advanced Trainer comparison cells
also stay on CPU. FineTuneLab forwards this setting explicitly as Trainer's
`use_cpu` option: the installed Transformers version otherwise rediscovers MPS
and can mix CPU batches with MPS model weights. All ten notebooks use tiny, random
official Qwen fixtures in this
mode and require no model downloads. Use `.venv/bin/...` or the activated shell to
keep the installed Mac environment; an unqualified `uv run` may sync a different
set of extras.

## Download the requested checkpoints

```bash
.venv/bin/python scripts/download_qwen35.py \
  --destination /Users/haroldye/Desktop/Code/models
```

The script resumes downloads and verifies all files, using SHA-256 for large-file
objects and Git blob hashes for ordinary repository files. It pins these revisions:

| Model | Revision | Snapshot size |
|---|---|---|
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4,571,274,023 bytes |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9,342,907,469 bytes |

Each snapshot lives in its own model-named subdirectory. Adjacent
`Qwen3.5-2B.verification.json` and `Qwen3.5-4B.verification.json` record revisions,
file sizes, and verified hashes. The public repositories do not require login.

These are post-trained Transformers checkpoints. `local_pretrained` now supports
CPU, MPS, and CUDA selection. See [device selection](DEVICES.md) for the bounded
2B LoRA training setup on MPS. DAPT on real weights requires a separate Base
checkpoint, which this setup does not download.

## Verify the installation

```bash
uv pip check --python .venv/bin/python
.venv/bin/ftlab doctor
.venv/bin/ruff check .
.venv/bin/mypy src
ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/pytest -m 'not gpu' -v
```

The test suite executes every notebook in an isolated kernel, checks training and
artifact reports, and never fetches model weights. CUDA tests remain excluded.
`ftlab doctor` reports both CUDA and MPS availability. You can also check MPS with:

```bash
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'
```

For real-model inference, run one process per checkpoint, sequentially:

```bash
.venv/bin/python scripts/smoke_mac_model.py ../models/Qwen3.5-2B \
  --output outputs/mac-validation/2b-mps.json
.venv/bin/python scripts/smoke_mac_model.py ../models/Qwen3.5-4B \
  --output outputs/mac-validation/4b-mps.json
```

These checks load local files only, use BF16 and eager attention, and generate four
tokens for text and a small synthetic image. They disable asynchronous weight
loading (`HF_DEACTIVATE_ASYNC_LOAD=1`): concurrent 4B weight conversions triggered
a native MPS shader initialization crash during initial validation. They record
results or Python exceptions as JSON; a native process crash must be diagnosed
from its exit status and log. Unsupported MPS operations may use PyTorch CPU fallback; an explicit
`--device cpu` rerun uses BF16 on CPU. Memory protections remain enabled. Successful
short inference does not establish that full fine-tuning, long contexts, or
simultaneous student/teacher models fit in 16 GB.

See `MAC_VALIDATION.md` for the recorded setup results.
