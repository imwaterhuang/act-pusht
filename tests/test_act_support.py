"""ACT support contracts independent of the student-owned training loop."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from mini_wam.data import ActDataset
from mini_wam.evaluation.act import act_checkpoint_score, generate_act_scene_split
from mini_wam.evaluation.scenes import load_scene_split
from mini_wam.training.act_config import load_act_config
from mini_wam.training.act_data import build_act_training_data


ROOT = Path(__file__).resolve().parents[1]


def test_full_data_and_boundary_actions_match_original_rows() -> None:
    if not (ROOT / "data/lerobot/pusht_image").exists():
        pytest.skip("Local Push-T data is unavailable")
    config = load_act_config(ROOT / "configs/act_smoke.yaml")
    data = build_act_training_data(config)
    dataset: ActDataset = data.dataset
    assert data.manifest["episode_count"] == 206
    assert len(dataset) == 25_444
    assert not hasattr(data, "validation_loader")
    episode_id, local_t, global_t, end = dataset.source_indices(0)
    assert local_t == 0
    first = dataset[0]
    assert first["observation_image"].shape == (3, 96, 96)
    assert first["agent_position"].shape == (2,)
    assert first["action_chunk"].shape == (16, 2)
    assert first["action_valid_mask"].shape == (16,)
    torch.testing.assert_close(
        data.normalization.denormalize_action(first["action_chunk"][0]),
        dataset.source[global_t]["action"].float(), atol=1e-4, rtol=0,
    )
    tail_index = end - global_t - 2
    tail = dataset[tail_index]
    assert dataset.source_indices(tail_index)[2] == end - 2
    assert int(tail["action_valid_mask"].sum()) == 1
    assert torch.count_nonzero(tail["action_chunk"][1:]) == 0
    assert tail["episode_id"].item() == episode_id


def test_config_rejects_mismatched_action_horizon(tmp_path: Path) -> None:
    config = copy.deepcopy(load_act_config(ROOT / "configs/act_smoke.yaml"))
    config["data"]["action_horizon"] = 8
    path = tmp_path / "bad.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="action_horizon"):
        load_act_config(path)
    assert act_checkpoint_score({"success_count": 3, "mean_final_coverage": 0.5}, 20) > (
        act_checkpoint_score({"success_count": 2, "mean_final_coverage": 1.0}, 10)
    )


def test_development_and_review_scenes_are_disjoint(tmp_path: Path) -> None:
    development_path = tmp_path / "development.json"
    review_path = tmp_path / "review.json"
    generate_act_scene_split(
        development_path, split="development", generation_seed=77, count=1,
    )
    generate_act_scene_split(
        review_path, split="review", generation_seed=77, count=1,
        excluded_paths=(development_path,),
    )
    development = load_scene_split(development_path, "development")
    review = load_scene_split(review_path, "review")
    assert development["scenes"][0]["state"] != review["scenes"][0]["state"]
    assert development["scenes"][0]["environment_seed"] != review["scenes"][0]["environment_seed"]
    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert saved["environment_version"]
