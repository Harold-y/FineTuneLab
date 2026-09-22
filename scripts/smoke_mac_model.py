"""Check a local Qwen checkpoint in one process; no downloads or training."""

import argparse
import json
import os
from pathlib import Path

# Must be set before importing torch. Keep the default MPS memory protections.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
# Concurrent dtype conversions while loading 4B weights can crash native MPS
# shader initialization. Sequential loading also reduces transient memory use.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from finetunelab.education import inspect_local_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"checkpoint": str(args.checkpoint.resolve()), "device": args.device, "cases": []}
    try:
        inspect_local_checkpoint(args.checkpoint)
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; rerun with --device cpu")
        print(f"Loading {args.checkpoint.name} on {args.device} in BF16", flush=True)
        processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.checkpoint,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map={"": args.device},
        ).eval()
        for with_image in (False, True):
            content = [{"type": "text", "text": "Name one color."}]
            if with_image:
                content.insert(0, {"type": "image"})
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            kwargs = {"images": [Image.new("RGB", (64, 64), "red")]} if with_image else {}
            batch = processor(text=[prompt], return_tensors="pt", **kwargs).to(args.device)
            print(f"Generating {'image + text' if with_image else 'text'}", flush=True)
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=4, do_sample=False)
            new_ids = generated[0, batch["input_ids"].shape[-1] :].cpu()
            if not len(new_ids):
                raise AssertionError("No generated tokens")
            case = {
                "input": "image + text" if with_image else "text",
                "tokens": len(new_ids),
                "decoded": processor.decode(new_ids, skip_special_tokens=True),
            }
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
