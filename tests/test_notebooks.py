"""Execute lessons in isolated kernels, not merely validate their JSON."""

import json
import os
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.integration
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_executes_offline(path: Path, tmp_path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(notebook, km=manager, timeout=240)
    environment = os.environ | {
        "FTLAB_NOTEBOOK_MODE": "tiny_cpu",
        "FTLAB_NOTEBOOK_OUTPUT": str(tmp_path),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "JUPYTER_PLATFORM_DIRS": "1",
        "HF_HOME": str(tmp_path / "hf-cache"),
        "HF_DATASETS_CACHE": str(tmp_path / "hf-cache" / "datasets"),
        "IPYTHONDIR": str(tmp_path / "ipython"),
    }
    try:
        client.execute(cwd=str(ROOT), env=environment)
    finally:
        nbformat.write(notebook, tmp_path / path.name)
        # An explicitly supplied KernelManager belongs to the caller.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    reports = list(tmp_path.glob("*/lesson_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    if path.name[:2] in {"01", "02", "03", "04", "05", "06", "07", "08"}:
        assert report["reload_tokens_equal"]
