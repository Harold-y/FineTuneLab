"""Download and verify the two pinned, official Hugging Face snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REVISIONS = {
    "Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[2] / "models"
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, revision in REVISIONS.items():
        repo_id = f"Qwen/{name}"
        directory = args.destination / name
        metadata = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        print(f"Downloading {repo_id} at {revision}", flush=True)
        snapshot_download(repo_id, revision=revision, local_dir=directory, max_workers=4)
        files = []
        for entry in metadata.siblings or []:
            path = directory / entry.rfilename
            size = path.stat().st_size
            if size != entry.size:
                raise ValueError(f"Size mismatch: {path}: {size} != {entry.size}")
            digest = hashlib.sha256() if entry.lfs else hashlib.sha1()
            if not entry.lfs:
                digest.update(f"blob {size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
            if digest.hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {path}")
            files.append({"file": entry.rfilename, "bytes": size, "hash": expected})
        for index in directory.glob("*.index.json"):
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if any(not (directory / shard).is_file() for shard in shards):
                raise ValueError(f"Incomplete shard index: {index}")
        manifest = {"repo_id": repo_id, "revision": revision, "verified": True, "files": files}
        (args.destination / f"{name}.verification.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(
            f"Verified {name}: {len(files)} files, {sum(x['bytes'] for x in files):,} bytes",
            flush=True,
        )


if __name__ == "__main__":
    main()
