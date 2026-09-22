import json
from pathlib import Path

from finetunelab.config import load_config
from finetunelab.runtime import write_manifest

ROOT = Path(__file__).parents[1]


def test_manifest_records_reproducibility_data(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen35/2b/sft_qlora.yaml")
    path = write_manifest(tmp_path, config, dataset_id="abc", parameters={"trainable": 10})
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["dataset_fingerprint"] == "abc"
    assert value["environment"]["python"]
    assert value["parameters"]["trainable"] == 10
