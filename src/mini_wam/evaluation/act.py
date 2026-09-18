"""ACT random-scene generation, closed-loop rollouts and checkpoint scoring."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from mini_wam.data import NormalizationStats
from mini_wam.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from mini_wam.evaluation.scenes import load_scene_split
from mini_wam.models.act import ActionPolicy
from mini_wam.studio.pusht import SceneState, make_pusht_env, random_scene, validate_scene

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _hash_records(records: list[dict[str, Any]]) -> str:
    raw = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"Refusing to replace frozen scenes: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _scene_key(record: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in SceneState(**record["state"]).as_array())


def generate_act_scene_split(
    path: str | Path,
    *,
    split: str,
    generation_seed: int,
    count: int,
    excluded_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Freeze one pool; create the review pool only after model selection."""
    if split not in ("development", "review") or count < 1:
        raise ValueError("split must be development/review and count must be positive")
    used_states: set[tuple[float, ...]] = set()
    used_seeds: set[int] = set()
    for excluded in excluded_paths:
        for record in load_scene_split(excluded)["scenes"]:
            used_states.add(_scene_key(record))
            used_seeds.add(int(record["environment_seed"]))
    rng = np.random.default_rng(generation_seed)
    scenes: list[dict[str, Any]] = []
    attempts = 0
    while len(scenes) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("Could not generate enough unique legal scenes")
        seed = int(rng.integers(0, 2**31 - 1))
        if seed in used_seeds:
            continue
        scene = random_scene(seed)
        try:
            validate_scene(scene)
        except ValueError:
            continue
        record = {"environment_seed": seed, "state": asdict(scene)}
        key = _scene_key(record)
        if key in used_states:
            continue
        used_states.add(key)
        used_seeds.add(seed)
        scenes.append({"scene_id": f"act-{split}-{len(scenes):03d}", **record})
    payload = {
        "schema_version": 1,
        "task": "pusht",
        "split": split,
        "generation_seed": generation_seed,
        "generator": "mini_wam.studio.pusht.random_scene",
        "environment_version": importlib.metadata.version("gym-pusht"),
        "count": len(scenes),
        "scenes_hash": _hash_records(scenes),
        "scenes": scenes,
    }
    _write_once(Path(path), payload)
    return payload


def _observation_tensors(
    observation: dict[str, np.ndarray],
    normalization: NormalizationStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.fromarray(np.asarray(observation["pixels"], dtype=np.uint8))
    image = image.resize((96, 96), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    pixels = ((pixels - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
    position = torch.as_tensor(observation["agent_pos"], dtype=torch.float32).reshape(1, 2)
    return pixels.to(device), normalization.normalize_position(position).to(device)


def _wilson_95(success_count: int, episode_count: int) -> list[float]:
    z = 1.959963984540054
    p = success_count / episode_count
    denominator = 1 + z * z / episode_count
    center = (p + z * z / (2 * episode_count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / episode_count + z * z / (4 * episode_count**2)) / denominator
    return [center - margin, center + margin]


def act_checkpoint_score(summary: dict[str, Any], step: int) -> tuple[int, float, int]:
    """Higher is better: strict successes, final coverage, then earlier step."""
    return (int(summary["success_count"]), float(summary["mean_final_coverage"]), -step)


def load_act_checkpoint(
    checkpoint_path: str | Path,
    audit_path: str | Path,
    device: torch.device,
) -> tuple[ActionPolicy, NormalizationStats, dict[str, Any]]:
    """Load only a matching ACT checkpoint with its recorded normalization."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint needs a config mapping")
    config = payload["config"]
    # Validate the essential serialized model contract before constructing it.
    if config.get("model", {}).get("name") != "act":
        raise ValueError("Checkpoint is not an ACT policy")
    metadata = payload.get("metadata", {})
    if metadata.get("architecture") != "act_v1":
        raise ValueError("Checkpoint architecture must be act_v1")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if metadata.get("dataset_fingerprint") != audit.get("dataset_fingerprint"):
        raise ValueError("Checkpoint and ACT audit use different data")
    saved_normalization = metadata.get("normalization")
    if not isinstance(saved_normalization, dict):
        raise ValueError("Checkpoint has no normalization copy")
    expected = audit["training_normalization"]
    for name, prefix in (("agent_position", "position"), ("action", "action")):
        for statistic in ("mean", "std"):
            value = saved_normalization.get(f"{prefix}_{statistic}")
            if not isinstance(value, list) or len(value) != 2 or not np.allclose(
                value, expected[name][statistic], rtol=1e-6, atol=1e-6
            ):
                raise ValueError("Checkpoint normalization differs from ACT audit")
    if saved_normalization.get("std_floor") != expected["std_floor"]:
        raise ValueError("Checkpoint normalization floor differs from ACT audit")
    normalization = NormalizationStats(
        **{
            key: torch.tensor(value, dtype=torch.float32) if isinstance(value, list) else value
            for key, value in saved_normalization.items()
        }
    )
    model_fields = (
        "pretrained", "d_model", "nhead", "num_encoder_layers",
        "num_latent_layers", "latent_dim", "chunk_size",
    )
    options = {key: config["model"][key] for key in model_fields}
    options["pretrained"] = False  # weights come only from this checkpoint
    model = ActionPolicy(**options)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), normalization, payload


def evaluate_act_policy(
    model: torch.nn.Module,
    normalization: NormalizationStats,
    scene_path: str | Path,
    output_dir: str | Path,
    *,
    device: torch.device,
    expected_split: str = "development",
    max_steps: int = 300,
    execute_steps: int = 4,
    save_videos: bool = False,
    success_coverage: float = 0.87,
) -> dict[str, Any]:
    """Run one ACT checkpoint on a pre-frozen scene file and save full traces."""
    if expected_split not in ("development", "review"):
        raise ValueError("Only ACT development or fresh review scenes are allowed")
    if not 0 <= success_coverage <= 1:
        raise ValueError("success_coverage must be between 0 and 1")
    scenes = load_scene_split(scene_path, expected_split=expected_split)
    if max_steps < 1 or execute_steps < 1:
        raise ValueError("max_steps and execute_steps must be positive")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if save_videos:
        (output / "videos").mkdir(exist_ok=True)
    was_training = model.training
    model.eval()
    episode_results: list[dict[str, Any]] = []
    try:
        for record in scenes["scenes"]:
            scene = SceneState(**{key: float(value) for key, value in record["state"].items()})
            env = make_pusht_env(max_steps=max_steps)
            frames: list[np.ndarray] = []
            trace: list[dict[str, Any]] = []
            inference_ms: list[float] = []
            reward_sum = max_coverage = final_coverage = 0.0
            steps = model_calls = 0
            terminated = truncated = strict_success = False
            try:
                observation, info = env.reset(
                    seed=int(record["environment_seed"]),
                    options={"reset_to_state": scene.as_array()},
                )
                if save_videos:
                    frames.append(np.asarray(env.render(), dtype=np.uint8))
                while steps < max_steps and not (terminated or truncated or strict_success):
                    image, position = _observation_tensors(observation, normalization, device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        predicted = model(image, position)
                    inference_ms.append((time.perf_counter() - started) * 1000.0)
                    model_calls += 1
                    if predicted.ndim != 3 or predicted.shape[0] != 1 or predicted.shape[2] != 2:
                        raise ValueError("ACT prediction must have shape [1, K, 2]")
                    actions = normalization.denormalize_action(predicted.detach().cpu())
                    actions = actions.squeeze(0).clamp(0, 512).numpy().astype(np.float32)
                    if execute_steps > len(actions):
                        raise ValueError("execute_steps exceeds model action chunk")
                    for action in actions[:execute_steps]:
                        if steps >= max_steps:
                            break
                        observation, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        reward_sum += float(reward)
                        final_coverage = float(info.get("coverage", 0.0))
                        max_coverage = max(max_coverage, final_coverage)
                        strict_success = final_coverage > success_coverage
                        simulator_success = bool(info.get("is_success", False))
                        trace.append({
                            "step": steps,
                            "action": action.tolist(),
                            "agent_position": np.asarray(observation["agent_pos"]).tolist(),
                            "coverage": final_coverage,
                            "simulator_success": simulator_success,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        })
                        if save_videos:
                            frames.append(np.asarray(env.render(), dtype=np.uint8))
                        if strict_success or terminated or truncated:
                            break
            finally:
                env.close()
            video_path = output / "videos" / f"{record['scene_id']}.mp4"
            if save_videos:
                if importlib.util.find_spec("av") is not None:
                    iio.imwrite(video_path, np.stack(frames), plugin="pyav", fps=10,
                                codec="libx264", out_pixel_format="yuv420p")
                else:
                    iio.imwrite(video_path, np.stack(frames), plugin="FFMPEG", fps=10,
                                codec="libx264", pixelformat="yuv420p")
            episode_results.append({
                "scene_id": record["scene_id"],
                "environment_seed": record["environment_seed"],
                "is_success": strict_success,
                "final_coverage": final_coverage,
                "max_coverage": max_coverage,
                "reward": reward_sum,
                "steps": steps,
                "model_calls": model_calls,
                "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
                "termination_reason": (
                    "strict_success" if strict_success else
                    "simulator_terminated" if terminated else
                    "time_limit" if truncated or steps >= max_steps else "unknown"
                ),
                "trace": trace,
                "video": str(video_path.relative_to(output)) if save_videos else None,
            })
    finally:
        model.train(was_training)
    success_count = sum(item["is_success"] for item in episode_results)
    count = len(episode_results)
    summary = {
        "episode_count": count,
        "success_count": success_count,
        "success_rate": success_count / count,
        "success_rate_wilson_95": _wilson_95(success_count, count),
        "mean_final_coverage": float(np.mean([item["final_coverage"] for item in episode_results])),
        "mean_max_coverage": float(np.mean([item["max_coverage"] for item in episode_results])),
        "mean_reward": float(np.mean([item["reward"] for item in episode_results])),
    }
    result = {
        "schema_version": 1,
        "policy": "act_v1",
        "scene_file": str(Path(scene_path).resolve()),
        "scenes_hash": scenes["scenes_hash"],
        "protocol": {
            "split": expected_split, "max_steps": max_steps,
            "execute_steps": execute_steps, "strict_success_coverage": success_coverage,
            "success_definition": "coverage > threshold; stop at first crossing",
            "videos_saved": save_videos,
        },
        "summary": summary,
        "episodes": episode_results,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
