"""Check the web adapter against the formal ACT inference path."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mini_wam.evaluation.act import (
    _observation_tensors,
    evaluate_act_policy,
    generate_act_scene_split,
    load_act_checkpoint,
)
from mini_wam.studio.checkpoints import checkpoint_normalization, load_checkpoint
from mini_wam.studio.pusht import SceneState, prepare_model_input, run_rollout

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "artifacts/studio/models/act_step_30000.pt"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Downloaded ACT weights are not available")
def test_web_act_matches_formal_inference_and_rollout(tmp_path):
    torch.set_num_threads(4)
    device = torch.device("cpu")
    reference, stats, _ = load_act_checkpoint(WEIGHTS, ROOT / "artifacts/act/data_audit.json", device)
    web_model, info = load_checkpoint(WEIGHTS, device)
    assert info.architecture == "act_v1" and info.step == 30000
    web_stats = checkpoint_normalization(WEIGHTS, stats, "builtin", "builtin")
    old = {"pixels": np.zeros((96, 96, 3), dtype=np.uint8), "agent_pos": np.array([10, 20])}
    current = {"pixels": np.full((96, 96, 3), 127, dtype=np.uint8), "agent_pos": np.array([230, 270])}
    images, positions = prepare_model_input([old, current], web_stats, device)
    image, position = _observation_tensors(current, stats, device)
    with torch.inference_mode():
        expected = reference(image, position)
        actual = web_model(images, positions)
        images[:, 0] = 99
        positions[:, 0] = -99
        changed_history = web_model(images, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(changed_history, expected, rtol=0, atol=0)
    scenes = generate_act_scene_split(tmp_path / "scenes.json", split="review", generation_seed=123, count=1)
    reference_result = evaluate_act_policy(reference, stats, tmp_path / "scenes.json", tmp_path / "formal", device=device, expected_split="review", max_steps=8)
    scene = SceneState(**scenes["scenes"][0]["state"])
    web_result = list(run_rollout(web_model, web_stats, scene, tmp_path / "web.mp4", max_steps=8))[-1]
    episode = reference_result["episodes"][0]
    assert web_result.metrics.steps == episode["steps"]
    assert web_result.metrics.model_calls == episode["model_calls"]
    assert web_result.metrics.final_coverage == pytest.approx(episode["final_coverage"], abs=1e-8)
    assert web_result.metrics.max_coverage == pytest.approx(episode["max_coverage"], abs=1e-8)
    assert web_result.video_path.is_file()
