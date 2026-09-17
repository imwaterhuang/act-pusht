"""ACT-specific configuration; legacy two-frame validators remain unchanged."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .config import resolve_project_path


def _required(config: dict[str, Any], path: str, kind: type) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"ACT config missing {path}")
        value = value[key]
    if kind is int and (type(value) is not int):
        raise TypeError(f"{path} must be an integer")
    if kind is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise TypeError(f"{path} must be a number")
    if kind not in (int, float) and not isinstance(value, kind):
        raise TypeError(f"{path} must be {kind.__name__}")
    return value


def load_act_config(path: str | Path) -> dict[str, Any]:
    """Validate structure and cross-field contracts without starting training."""
    config = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("ACT config top level must be a mapping")
    if _required(config, "model.name", str) != "act":
        raise ValueError("model.name must be act")
    _required(config, "model.pretrained", bool)
    for name in ("d_model", "nhead", "num_encoder_layers", "num_latent_layers",
                 "latent_dim", "chunk_size"):
        if _required(config, f"model.{name}", int) <= 0:
            raise ValueError(f"model.{name} must be positive")
    if config["model"]["d_model"] % config["model"]["nhead"]:
        raise ValueError("model.d_model must be divisible by model.nhead")
    if config["model"]["d_model"] % 4:
        raise ValueError("model.d_model must be divisible by 4 for spatial encodings")
    for name in ("dataset_root", "episode_manifest", "normalization_path"):
        _required(config, f"data.{name}", str)
    horizon = _required(config, "data.action_horizon", int)
    if horizon != config["model"]["chunk_size"]:
        raise ValueError("data.action_horizon must equal model.chunk_size")
    _required(config, "training.seed", int)
    for name in ("batch_size", "train_steps"):
        if _required(config, f"training.{name}", int) <= 0:
            raise ValueError(f"training.{name} must be positive")
    warmup = _required(config, "training.warmup_steps", int)
    if not 0 <= warmup < config["training"]["train_steps"]:
        raise ValueError("training.warmup_steps must be in [0, train_steps)")
    for name in ("learning_rate", "gradient_clip_norm"):
        value = _required(config, f"training.{name}", float)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be positive")
    for name in ("weight_decay", "beta"):
        value = _required(config, f"training.{name}", float)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} must be nonnegative")
    for name in ("frequency",):
        if _required(config, f"checkpoint.{name}", int) <= 0:
            raise ValueError(f"checkpoint.{name} must be positive")
    archive = config["checkpoint"].get("archive_frequency")
    if archive is not None and (
        type(archive) is not int or archive < config["checkpoint"]["frequency"]
        or archive % config["checkpoint"]["frequency"]
    ):
        raise ValueError("checkpoint.archive_frequency must be a frequency multiple")
    enabled = _required(config, "evaluation.enabled", bool)
    if enabled:
        _required(config, "evaluation.development_scenes", str)
        for name in ("frequency", "max_steps", "execute_steps"):
            if _required(config, f"evaluation.{name}", int) <= 0:
                raise ValueError(f"evaluation.{name} must be positive")
        if config["evaluation"]["execute_steps"] > horizon:
            raise ValueError("evaluation.execute_steps cannot exceed action horizon")
    return config


def act_asset_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve data paths relative to the project, not the current shell."""
    result = {
        key: resolve_project_path(config["data"][key])
        for key in ("dataset_root", "episode_manifest", "normalization_path")
    }
    if config["evaluation"]["enabled"]:
        result["development_scenes"] = resolve_project_path(
            config["evaluation"]["development_scenes"]
        )
    return result
