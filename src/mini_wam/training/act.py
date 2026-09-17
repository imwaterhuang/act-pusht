"""ACT (Action Chunking with Transformers) training and checkpoint selection."""

from __future__ import annotations

import json
import csv
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from mini_wam.data import NormalizationStats
from mini_wam.evaluation.act import act_checkpoint_score, evaluate_act_policy
from mini_wam.evaluation.scenes import load_scene_split
from mini_wam.models.act import ActionPolicy, act_loss

from .act_config import act_asset_paths, load_act_config
from .act_data import build_act_training_data
from .artifacts import (
    CHECKPOINT_VERSION, append_csv, atomic_torch_save, default_run_dir,
    environment_summary, normalization_dict, sha256_file, log, sync_run_directory,
)
from .reproducibility import (
    DeterministicBatchSampler, capture_random_states, restore_random_states,
    seed_everything,
)
from .runtime import select_device
from .steps import make_scheduler

def train_act(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    stop_after_step: int | None = None,
    device_name: str = "auto",
    num_workers: int = 0,
    mirror_dir: str | Path | None = None,
) -> Path:
    """Train ACT on every demonstration and select checkpoints on frozen scenes."""
    if stop_after_step is not None and (
        type(stop_after_step) is not int or stop_after_step <= 0
    ):
        raise ValueError("stop_after_step must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")

    config_path = Path(config_path).expanduser().resolve()
    resume = Path(resume_path).expanduser().resolve() if resume_path is not None else None
    run_path = Path(run_dir).expanduser().resolve() if run_dir is not None else None
    mirror_path = Path(mirror_dir).expanduser().resolve() if mirror_dir is not None else None
    if resume is not None:
        if run_path is not None and run_path != resume.parent.parent:
            raise ValueError("run_dir must match the resumed checkpoint's run directory")
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    elif run_path is not None and run_path.exists():
        if not run_path.is_dir() or any(run_path.iterdir()):
            raise ValueError("A new run requires an empty or nonexistent run_dir")

    config = load_act_config(config_path)
    if stop_after_step is not None and stop_after_step > config["training"]["train_steps"]:
        raise ValueError("stop_after_step must not exceed training.train_steps")

    training = config["training"]
    device = select_device(device_name)
    seed_everything(int(training["seed"]))

    asset_paths = act_asset_paths(config)
    data = build_act_training_data(config, num_workers=num_workers, verify_source=True)
    if data.manifest["episode_count"] != 206:
        raise ValueError("ACT training requires all 206 demonstration episodes")
    train_loader = data.train_loader
    sampler = data.sampler
    stats = data.normalization
    manifest = data.manifest
    dataset_fingerprint = manifest["dataset_fingerprint"]
    experiment_fingerprints = {
        "dataset_fingerprint": dataset_fingerprint,
        "manifest_sha256": sha256_file(asset_paths["episode_manifest"]),
        "normalization_sha256": sha256_file(asset_paths["normalization_path"]),
    }

    model = ActionPolicy(
        **{key: value for key, value in config["model"].items() if key != "name"}
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = make_scheduler(
        optimizer,
        int(training["warmup_steps"]),
        int(training["train_steps"]),
    )

    step = 0
    target_step = stop_after_step or int(training["train_steps"])
    if resume is not None:
        run_path = resume.parent.parent
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Incompatible ACT checkpoint version")
        metadata = checkpoint.get("metadata", {})
        if metadata.get("architecture") != "act_v1":
            raise ValueError("Resume requires an act_v1 checkpoint")
        saved_config = checkpoint.get("config", {})
        if (saved_config.get("model") != config["model"]
            or saved_config.get("training") != training
            or saved_config.get("data", {}).get("action_horizon") != config["data"]["action_horizon"]):
            raise ValueError("Resume model or training settings differ from the checkpoint")
        if metadata.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Resume dataset fingerprint differs from the checkpoint")
        step = checkpoint["step"]
        if type(step) is not int or not 0 <= step < target_step:
            raise ValueError("Checkpoint step must be nonnegative and before target_step")
        if (checkpoint["scheduler"] is None) != (scheduler is None):
            raise ValueError("Resume scheduler differs from the checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.load_state_dict(checkpoint["sampler"])
        restore_random_states(checkpoint["random_states"])
    else:
        if run_path is None:
            run_path = default_run_dir(int(training["seed"]), model_name="act")
        if run_path.exists() and (not run_path.is_dir() or any(run_path.iterdir())):
            raise ValueError("A new run requires an empty or nonexistent run_dir")
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "checkpoints").mkdir()
        shutil.copy2(config_path, run_path / "config.yaml")
        shutil.copy2(asset_paths["episode_manifest"], run_path / "episode_manifest.json")
        shutil.copy2(asset_paths["normalization_path"], run_path / "normalization.json")
        (run_path / "experiment_fingerprints.json").write_text(
            json.dumps(experiment_fingerprints, indent=2), encoding="utf-8"
        )
        environment = environment_summary(device)
        environment["architecture"] = "act_v1"
        environment["data_loader"] = {"train_num_workers": num_workers}
        (run_path / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if run_path is None:
        raise RuntimeError("ACT run directory was not initialized")
    if mirror_path is not None and mirror_path != run_path and mirror_path.is_relative_to(run_path):
        raise ValueError("mirror_dir must be outside the run directory")
    elapsed_before = 0.0
    metrics_path = run_path / "train_metrics.csv"
    if resume is not None and metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        kept = [row for row in rows if int(row["step"]) <= step]
        if len(kept) != len(rows):
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(kept)
        if kept:
            elapsed_before = float(kept[-1]["elapsed_seconds"])

    selection_path = run_path / "selection.json"
    selection: dict[str, Any] = {"candidates": [], "best_step": None}
    if resume is not None and selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["candidates"] = [
            candidate for candidate in selection["candidates"]
            if candidate["step"] <= step
        ]
        if selection["candidates"]:
            selection["best_step"] = max(
                selection["candidates"],
                key=lambda item: act_checkpoint_score(item["summary"], item["step"]),
            )["step"]
        else:
            selection["best_step"] = None
            (run_path / "checkpoints" / "best.pt").unlink(missing_ok=True)
        if selection["best_step"] is not None:
            best_record = next(
                item for item in selection["candidates"]
                if item["step"] == selection["best_step"]
            )
            saved_best = run_path / best_record["checkpoint"]
            if saved_best.exists():
                shutil.copy2(saved_best, run_path / "checkpoints" / "best.pt")
        selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    if config["evaluation"]["enabled"] and selection["candidates"]:
        scene_hash = load_scene_split(
            asset_paths["development_scenes"], expected_split="development"
        )["scenes_hash"]
        saved_evaluation = checkpoint["config"]["evaluation"]
        if scene_hash != selection["candidates"][0]["scenes_hash"] or any(
            saved_evaluation[key] != config["evaluation"][key]
            for key in ("max_steps", "execute_steps")
        ):
            raise ValueError("Development scenes or rollout protocol changed during checkpoint selection")
    if resume is not None:
        (run_path / "completed.json").unlink(missing_ok=True)

    log(run_path, f"ACT run: step {step} -> {target_step}, device={device}")
    started = time.perf_counter()
    batches = iter(train_loader)
    for step in range(step + 1, target_step + 1):
        step_started = time.perf_counter()
        batch = next(batches)
        metrics = train_act_step(
            model=model, batch=batch, optimizer=optimizer, scheduler=scheduler,
            sampler=sampler, device=device, config=config,
        )
        clean_stop = step == target_step
        checkpoint_path = log_act_step(
            run_dir=run_path, model=model, optimizer=optimizer,
            scheduler=scheduler, sampler=sampler, step=step, config=config,
            stats=stats, dataset_fingerprint=dataset_fingerprint,
            loss=metrics["loss"], action_l1=metrics["action_l1"],
            kl=metrics["kl"], gradient_norm=metrics["gradient_norm"],
            step_seconds=time.perf_counter() - step_started,
            elapsed_seconds=elapsed_before + time.perf_counter() - started,
            clean_stop=clean_stop,
        )
        if checkpoint_path is not None and mirror_path is not None:
            sync_run_directory(run_path, mirror_path)
        log(run_path, f"step={step} loss={metrics['loss']:.6f} action_l1={metrics['action_l1']:.6f} kl={metrics['kl']:.6f}")

        evaluation = config["evaluation"]
        evaluate_now = evaluation["enabled"] and (
            step % evaluation["frequency"] == 0
            or step == training["train_steps"]
        )
        if evaluate_now:
            if checkpoint_path is None:
                checkpoint_path = run_path / "checkpoints" / "last.pt"
                atomic_torch_save(
                    act_checkpoint_payload(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        sampler=sampler, step=step, config=config, stats=stats,
                        dataset_fingerprint=dataset_fingerprint,
                    ), checkpoint_path,
                )
            random_states = capture_random_states()
            candidate_path = run_path / "evaluation" / "candidates" / f"step_{step:07d}.pt"
            # Candidate weights are sufficient for evaluation; last.pt keeps
            # optimizer, sampler and random states for training recovery.
            atomic_torch_save({
                "model": model.state_dict(),
                "config": config,
                "step": step,
                "metadata": {
                    "architecture": "act_v1",
                    "dataset_fingerprint": dataset_fingerprint,
                    "normalization": normalization_dict(stats),
                },
            }, candidate_path)
            try:
                result = evaluate_act_policy(
                    model, stats, asset_paths["development_scenes"],
                    run_path / "evaluation" / f"step_{step:07d}",
                    device=device, max_steps=evaluation["max_steps"],
                    execute_steps=evaluation["execute_steps"],
                )
            finally:
                restore_random_states(random_states)
                model.train()
            summary = result["summary"]
            score = act_checkpoint_score(summary, step)
            previous = next(
                (item for item in selection["candidates"]
                 if item["step"] == selection["best_step"]), None,
            )
            is_best = previous is None or score > act_checkpoint_score(
                previous["summary"], previous["step"]
            )
            candidate = {
                "step": step, "checkpoint": str(candidate_path.relative_to(run_path)),
                "checkpoint_sha256": sha256_file(candidate_path),
                "scenes_hash": result["scenes_hash"], "summary": summary,
                "selected": is_best,
            }
            selection["candidates"].append(candidate)
            if is_best:
                shutil.copy2(candidate_path, run_path / "checkpoints" / "best.pt")
                selection["best_step"] = step
            selection_path.write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if mirror_path is not None:
                sync_run_directory(run_path, mirror_path)
            log(run_path, f"development step={step} success={summary['success_count']}/{summary['episode_count']} best_step={selection['best_step']}")

    if target_step == training["train_steps"]:
        completion = {
            "completed": True,
            "step": target_step,
            "checkpoint": "checkpoints/last.pt",
            "checkpoint_sha256": sha256_file(run_path / "checkpoints" / "last.pt"),
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        }
        (run_path / "completed.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        log(run_path, f"ACT training complete at step {target_step}")
    else:
        log(run_path, f"ACT clean stop at step {target_step}; target={training['train_steps']}")
    if mirror_path is not None:
        sync_run_directory(run_path, mirror_path)
    return run_path


def train_act_step(
    *,
    model: ActionPolicy,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    sampler: DeterministicBatchSampler,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    """Complete one update and return separate metrics for ``log_act_step``."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    image = batch["observation_image"].to(device)
    position = batch["agent_position"].to(device)
    actions = batch["action_chunk"].to(device)
    mask = batch["action_valid_mask"].to(device)
    output = model(image, position, actions, mask)
    losses = act_loss(
        output["actions"], actions, mask, output["mu"], output["logvar"],
        beta=float(config["training"]["beta"]),
    )
    if not torch.isfinite(torch.stack(list(losses.values()))).all():
        raise FloatingPointError("Non-finite ACT loss")
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["training"]["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    sampler.mark_consumed()
    metrics = {name: float(value.detach()) for name, value in losses.items()}
    metrics["gradient_norm"] = float(gradient_norm)
    return metrics


def act_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    sampler: DeterministicBatchSampler,
    step: int,
    config: dict[str, Any],
    stats: NormalizationStats,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    """Capture state at a completed optimizer step, after advancing the sampler.

    Serialize the result immediately: model and optimizer tensors are live views.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative completed optimizer step")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "sampler": sampler.state_dict(),
        "random_states": capture_random_states(),
        "step": step,
        "config": config,
        "metadata": {
            "architecture": "act_v1",
            "dataset_fingerprint": dataset_fingerprint,
            "normalization": normalization_dict(stats),
        },
    }


def log_act_step(
    *,
    run_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    sampler: DeterministicBatchSampler,
    step: int,
    config: dict[str, Any],
    stats: NormalizationStats,
    dataset_fingerprint: str,
    loss: float | torch.Tensor,
    action_l1: float | torch.Tensor,
    kl: float | torch.Tensor,
    gradient_norm: float | torch.Tensor,
    step_seconds: float,
    elapsed_seconds: float,
    clean_stop: bool = False,
) -> Path | None:
    """Record one completed update and save when due or explicitly stopping.

    Call after the optimizer, scheduler and sampler have advanced. Set
    clean_stop on the final update of an early stop; reaching train_steps saves
    automatically. Timings are measured by the caller around the training work.
    """
    if type(step) is not int or step <= 0:
        raise ValueError("step must be a positive completed optimizer step")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    row = {
        "step": step,
        "loss": float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss),
        "action_l1": float(action_l1.detach()) if isinstance(action_l1, torch.Tensor) else float(action_l1),
        "kl": float(kl.detach()) if isinstance(kl, torch.Tensor) else float(kl),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "gradient_norm": float(gradient_norm.detach()) if isinstance(gradient_norm, torch.Tensor) else float(gradient_norm),
        "step_seconds": float(step_seconds),
        "elapsed_seconds": float(elapsed_seconds),
    }
    append_csv(run_path / "train_metrics.csv", list(row), row)

    checkpoint_config = config["checkpoint"]
    final_step = step == config["training"]["train_steps"]
    if step % checkpoint_config["frequency"] and not clean_stop and not final_step:
        return None
    payload = act_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        step=step,
        config=config,
        stats=stats,
        dataset_fingerprint=dataset_fingerprint,
    )
    checkpoint_path = run_path / "checkpoints" / "last.pt"
    atomic_torch_save(payload, checkpoint_path)
    archive_frequency = checkpoint_config.get("archive_frequency")
    if archive_frequency is not None and (step % archive_frequency == 0 or final_step):
        atomic_torch_save(payload, checkpoint_path.with_name(f"step_{step:07d}.pt"))
    return checkpoint_path
