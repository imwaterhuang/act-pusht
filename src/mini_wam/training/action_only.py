"""ActionOnlyPolicy 的可复现训练总流程。

本模块只负责装配数据、模型、恢复、训练、验证和保存。具体职责位于：

- ``config``：配置校验与项目路径；
- ``reproducibility``：确定性采样与随机状态；
- ``steps``：单步优化、学习率调度与离线验证；
- ``artifacts``：指标、checkpoint（检查点）、实验身份和目录镜像。

文件末尾保留少量旧私有名称别名，兼容已有脚本与 notebook（交互式笔记本）。
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from mini_wam.models.action_only import ActionOnlyPolicy
from mini_wam.studio.datasets import validate_dataset

from .artifacts import (
    CHECKPOINT_VERSION,
    append_csv,
    atomic_torch_save,
    checkpoint_payload,
    environment_summary,
    log,
    normalization_dict,
    prepare_run_directory,
    sha256_file,
    sync_run_directory,
)
from .config import PROJECT_ROOT, load_training_config, resolve_project_path
from .data import build_training_data
from .reproducibility import (
    DeterministicBatchSampler,
    capture_random_states,
    restore_random_states,
    seed_everything,
)
from .runtime import select_device
from .steps import evaluate, make_scheduler, train_step


def train_action_only(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    mirror_dir: str | Path | None = None,
    stop_after_step: int | None = None,
    device_name: str = "auto",
    num_workers: int = 0,
) -> Path:
    """训练、验证、保存并可选恢复 ActionOnlyPolicy。"""
    config_path = Path(config_path).expanduser().resolve()
    config = load_training_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    train_steps = int(training["train_steps"])
    if stop_after_step is not None and not 1 <= stop_after_step <= train_steps:
        raise ValueError("stop_after_step 必须在 [1, train_steps] 内")
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    target_step = stop_after_step or train_steps
    device = select_device(device_name)

    resume, run_path, mirror_path = prepare_run_directory(
        seed=seed,
        model_name="action_only",
        resume_path=resume_path,
        run_dir=run_dir,
        mirror_dir=mirror_dir,
    )
    os.environ.setdefault(
        "HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface")
    )
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets")
    )
    seed_everything(seed)

    data = build_training_data(
        config=config,
        seed=seed,
        num_workers=num_workers,
        include_future_observations=False,
    )
    stats = data.stats
    dataset_root = data.dataset_root
    split_path = data.split_path
    sampler = data.sampler
    train_loader = data.train_loader
    validation_loader = data.validation_loader

    model = ActionOnlyPolicy(pretrained=bool(config["model"]["pretrained"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = make_scheduler(
        optimizer,
        int(training["warmup_steps"]),
        train_steps,
    )
    environment = environment_summary(device)
    environment["data_loader"] = {
        "train_num_workers": num_workers,
        "validation_num_workers": 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }
    split_hash = sha256_file(split_path)
    dataset_fingerprint = validate_dataset(
        dataset_root, "lerobot/pusht_image"
    ).fingerprint
    best_validation_loss = math.inf
    last_validation_loss: float | None = None
    step = 0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("checkpoint 版本不兼容")
        if checkpoint.get("config") != config:
            raise ValueError("恢复失败：当前配置与 checkpoint 配置不同")
        if checkpoint.get("split_hash") != split_hash:
            raise ValueError("恢复失败：数据划分已经变化")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.load_state_dict(checkpoint["sampler"])
        step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        last_validation_loss = checkpoint.get("validation_loss")
        restore_random_states(checkpoint["random_states"])
        if step >= target_step:
            raise ValueError(
                f"checkpoint 已在第 {step} 步，不早于目标第 {target_step} 步"
            )

    if not (run_path / "config.yaml").exists():
        shutil.copy2(config_path, run_path / "config.yaml")
    (run_path / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_path / "normalization.json").write_text(
        json.dumps(normalization_dict(stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        run_path,
        f"device={device} num_workers={num_workers} step={step} "
        f"target_step={target_step} run_dir={run_path}",
    )
    if mirror_path:
        sync_run_directory(run_path, mirror_path)

    checkpoint_frequency = int(config["checkpoint"]["frequency"])
    archive_frequency = int(
        config["checkpoint"].get("archive_frequency", checkpoint_frequency)
    )
    validation_frequency = int(config["validation"]["frequency"])
    max_validation_batches = config["validation"].get("max_batches")
    iterator = iter(train_loader)

    def current_checkpoint() -> dict[str, Any]:
        return checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            step=step,
            config=config,
            best_validation_loss=best_validation_loss,
            validation_loss=last_validation_loss,
            stats=stats,
            split_hash=split_hash,
            dataset_fingerprint=dataset_fingerprint,
            environment=environment,
        )

    while step < target_step:
        batch = next(iterator)
        loss_value, gradient_norm = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            device=device,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            next_step=step + 1,
        )
        step += 1
        append_csv(
            run_path / "train_metrics.csv",
            ["step", "loss", "learning_rate", "gradient_norm"],
            {
                "step": step,
                "loss": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
            },
        )

        if step % validation_frequency == 0 or step == target_step:
            last_validation_loss = evaluate(
                model,
                validation_loader,
                device,
                max_validation_batches,
            )
            append_csv(
                run_path / "val_metrics.csv",
                ["step", "validation_loss", "batches"],
                {
                    "step": step,
                    "validation_loss": last_validation_loss,
                    "batches": (
                        max_validation_batches
                        if max_validation_batches is not None
                        else "all"
                    ),
                },
            )
            log(
                run_path,
                f"step={step} train_loss={loss_value:.6f} "
                f"val_loss={last_validation_loss:.6f}",
            )
            if last_validation_loss < best_validation_loss:
                best_validation_loss = last_validation_loss
                atomic_torch_save(
                    current_checkpoint(), run_path / "checkpoints" / "best.pt"
                )
            if mirror_path:
                sync_run_directory(run_path, mirror_path)

        if step % checkpoint_frequency == 0 or step == target_step:
            payload = current_checkpoint()
            atomic_torch_save(payload, run_path / "checkpoints" / "last.pt")
            if step % archive_frequency == 0 or step == train_steps:
                atomic_torch_save(
                    payload,
                    run_path / "checkpoints" / f"step_{step:07d}.pt",
                )
            if mirror_path:
                sync_run_directory(run_path, mirror_path)

    log(run_path, f"completed target step {target_step}")
    if mirror_path:
        sync_run_directory(run_path, mirror_path)
    return run_path


# 兼容已有脚本和测试；新代码应直接从对应职责模块导入。
_resolve_project_path = resolve_project_path
_capture_random_states = capture_random_states
_restore_random_states = restore_random_states
_sync_run_directory = sync_run_directory
