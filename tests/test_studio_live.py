import numpy as np
import pytest
import torch

from mini_wam.data.dataset import NormalizationStats
from mini_wam.studio.live import LiveSession


class HoldPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, images, positions):
        self.inputs.append(images[:, -1].clone())
        return positions[:, -1, None].expand(-1, 16, -1) + self.anchor


def make_session():
    return LiveSession(HoldPolicy(), NormalizationStats(
        position_mean=torch.zeros(2), position_std=torch.ones(2),
        action_mean=torch.zeros(2), action_std=torch.ones(2),
    ))


def test_drag_reobserves_and_discards_pre_drag_actions():
    session = make_session()
    try:
        session.tick({})
        assert len(session.actions) == 3
        old_frame = session.model.inputs[-1]
        moved = session.tick({"drag": {"x": 350, "y": 350, "held": True}})
        assert moved["block"] == pytest.approx([350, 350])
        assert moved["moves"] == 1
        assert session.model_calls == 2
        assert not torch.equal(old_frame, session.model.inputs[-1])
        assert not session.actions  # held block was reobserved after physics too
        session.tick({"drag": {"x": 350, "y": 350, "held": False}})
        assert session.drag_target is None
        steps = session.steps
        paused = session.tick({"paused": True, "drag": {"x": 300, "y": 300, "held": False}})
        assert paused["steps"] == steps and paused["block"] == pytest.approx([300, 300])
        assert session.tick({"paused": False})["steps"] == steps + 1
        with pytest.raises(ValueError):
            session.move_block(float("nan"), 200)
    finally:
        session.close()


def test_live_continues_beyond_time_limit_and_success():
    session = make_session()
    try:
        session.steps = 300
        # Use the real simulator goal pose; the continuous loop must ignore its
        # terminated=True signal while preserving the body state and counters.
        session.env.block.angle = session.env.goal_pose[2]
        session.move_block(*session.env.goal_pose[:2])
        session.env.agent.position = (50, 50)
        session.history.clear()
        obs = session.env.get_obs()
        session.history.extend([obs, obs])
        first = session.tick({})
        second = session.tick({})
        assert first["coverage"] > .95 and second["coverage"] > .95
        assert first["steps"] == 301 and second["steps"] == 302
        assert session.env.block.position.x == pytest.approx(session.env.goal_pose[0])
    finally:
        session.close()
