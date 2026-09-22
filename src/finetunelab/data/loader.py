"""Load local files or Hub datasets and normalize image paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, Image, IterableDataset, List, load_dataset

from finetunelab.config import Modality, RecipeConfig
from finetunelab.data.validation import validate_dataset
from finetunelab.errors import DataValidationError


def _source_kind(config: RecipeConfig) -> str:
    if config.data.source_type != "auto":
        return config.data.source_type
    return "local" if Path(config.data.source).expanduser().exists() else "hub"


def _local_builder(config: RecipeConfig) -> tuple[str, dict[str, str]]:
    path = Path(config.data.source).expanduser().resolve()
    if config.data.data_files:
        root = path if path.is_dir() else path.parent
        files = {
            split: str((root / Path(filename).expanduser()).resolve())
            for split, filename in config.data.data_files.items()
        }
        for filename in files.values():
            if not Path(filename).is_file():
                raise DataValidationError(f"Local split file does not exist: {filename}")
        suffixes = {Path(filename).suffix.lower() for filename in files.values()}
        if ".parquet" in suffixes and len(suffixes) != 1:
            raise DataValidationError("Use one file format for all local splits")
        builder = config.data.format
        if builder == "auto":
            builder = "parquet" if suffixes == {".parquet"} else "json"
        return ("json" if builder == "jsonl" else builder), files
    if not path.exists():
        raise DataValidationError(f"Local dataset does not exist: {path}")
    suffix = path.suffix.lower()
    builder = config.data.format
    if builder == "auto":
        builder = "parquet" if suffix == ".parquet" else "json"
    if builder == "jsonl":
        builder = "json"
    return builder, {config.data.train_split: str(path)}


def _resolve_images(example: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve local image references while leaving PIL/HF Image objects untouched."""

    for field in ("image", "images"):
        if field not in example or example[field] is None:
            continue
        is_list = isinstance(example[field], list)
        values = example[field] if is_list else [example[field]]
        resolved: list[Any] = []
        for value in values:
            if isinstance(value, str) and not value.startswith(("http://", "https://")):
                candidate = Path(value)
                candidate = candidate if candidate.is_absolute() else root / candidate
                if not candidate.exists():
                    raise DataValidationError(f"Image does not exist: {candidate}")
                resolved.append(str(candidate.resolve()))
            else:
                resolved.append(value)
        example[field] = resolved if is_list else resolved[0]
    return example


def load_datasets(config: RecipeConfig) -> tuple[Dataset | IterableDataset, Any | None]:
    """Load train/eval splits and perform bounded schema validation."""

    if _source_kind(config) == "local":
        builder, files = _local_builder(config)
        raw = load_dataset(builder, data_files=files, streaming=config.data.streaming)
    else:
        raw = load_dataset(
            config.data.source,
            data_files=config.data.data_files,
            streaming=config.data.streaming,
        )

    if config.data.train_split not in raw:
        raise DataValidationError(
            f"Dataset has no train split named {config.data.train_split!r}; found {list(raw)}"
        )
    train = raw[config.data.train_split]
    if config.data.eval_split and config.data.eval_split not in raw:
        raise DataValidationError(
            f"Missing evaluation split {config.data.eval_split!r}. "
            "For local files provide data.data_files with train and validation entries."
        )
    evaluation = raw[config.data.eval_split] if config.data.eval_split else None

    if config.data.modality == Modality.IMAGE_TEXT:
        source = Path(config.data.source).expanduser().resolve()
        root = config.data.image_root or (source if source.is_dir() else source.parent)
        map_kwargs = {} if config.data.streaming else {"num_proc": config.data.num_proc}
        train = train.map(lambda row: _resolve_images(row, root), **map_kwargs)
        if "image" in train.column_names:
            train = train.cast_column("image", Image())
        elif "images" in train.column_names:
            train = train.cast_column("images", List(Image()))
        if evaluation is not None:
            evaluation = evaluation.map(lambda row: _resolve_images(row, root), **map_kwargs)
            if "image" in evaluation.column_names:
                evaluation = evaluation.cast_column("image", Image())
            elif "images" in evaluation.column_names:
                evaluation = evaluation.cast_column("images", List(Image()))

    validate_dataset(train, config)
    if evaluation is not None:
        validate_dataset(evaluation, config)
    return train, evaluation
