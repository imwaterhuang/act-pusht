"""Full-demonstration ACT data loading; no held-out expert validation loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_wam.data import ActDataset, NormalizationStats
from mini_wam.data.act_audit import build_act_audit

from .act_config import act_asset_paths
from .reproducibility import DeterministicBatchSampler


@dataclass(frozen=True)
class ActTrainingData:
    dataset: ActDataset
    train_loader: DataLoader
    sampler: DeterministicBatchSampler
    normalization: NormalizationStats
    manifest: dict[str, Any]
    audit: dict[str, Any]
    manifest_path: Path
    audit_path: Path


def build_act_training_data(
    config: dict[str, Any],
    *,
    num_workers: int = 0,
    image_cache_path: str | Path | None = None,
    verify_source: bool = True,
) -> ActTrainingData:
    """Load every listed episode and verify source identity before training."""
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    paths = act_asset_paths(config)
    manifest = json.loads(paths["episode_manifest"].read_text(encoding="utf-8"))
    audit = json.loads(paths["normalization_path"].read_text(encoding="utf-8"))
    ids = manifest.get("train")
    if (manifest.get("schema_version") != 1 or audit.get("schema_version") != 1
        or not isinstance(ids, list) or not ids
        or any(type(value) is not int for value in ids)
        or len(ids) != len(set(ids))
        or manifest.get("episode_count") != len(ids)
        or manifest.get("dataset_fingerprint") != audit.get("dataset_fingerprint")
        or manifest.get("valid_window_count") != audit.get("valid_window_count")):
        raise ValueError("ACT manifest and normalization audit are incompatible")
    if verify_source:
        actual_manifest, actual_audit = build_act_audit(paths["dataset_root"])
        if actual_manifest != manifest or actual_audit != audit:
            raise ValueError("ACT source data differs from the frozen manifest/statistics")
    normalization = NormalizationStats.from_audit_file(paths["normalization_path"])
    dataset = ActDataset(
        paths["dataset_root"], ids, normalization,
        action_horizon=config["data"]["action_horizon"],
        image_cache_path=image_cache_path,
    )
    if len(dataset) != manifest["valid_window_count"]:
        raise ValueError("ACT dataset window count differs from the frozen audit")
    seed = config["training"]["seed"]
    sampler = DeterministicBatchSampler(
        len(dataset), config["training"]["batch_size"], seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        generator=torch.Generator().manual_seed(seed + 100_000),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return ActTrainingData(
        dataset, loader, sampler, normalization, manifest, audit,
        paths["episode_manifest"], paths["normalization_path"],
    )
