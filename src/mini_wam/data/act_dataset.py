"""Single-frame Push-T windows for ACT (Action Chunking with Transformers)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import IMAGENET_MEAN, IMAGENET_STD, NormalizationStats


class ActDataset(Dataset[dict[str, Tensor]]):
    """Return current observation and up to ``action_horizon`` valid targets.

    Every episode contributes local steps 0 through length - 2. The final row
    has no within-episode next observation and never supplies an action label.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        episode_ids: list[int],
        normalization: NormalizationStats,
        action_horizon: int = 16,
        image_cache_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if not episode_ids or len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode_ids must be nonempty and unique")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.normalization = normalization
        self.action_horizon = action_horizon
        project_root = Path(__file__).resolve().parents[3]
        os.environ.setdefault("HF_HOME", str(project_root / "data/.cache/huggingface"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(project_root / "data/.cache/hf_datasets"))

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.source = LeRobotDataset(
            "lerobot/pusht_image", root=self.dataset_root, return_uint8=True
        )
        self._states, self._actions = self._load_scalar_tensors()
        self._windows = self._build_window_index(set(episode_ids))
        if not self._windows:
            raise ValueError("Selected episodes contain no valid action transitions")
        self._image_cache = None
        if image_cache_path is not None:
            from .image_cache import NormalizedImageCache

            self._image_cache = NormalizedImageCache(
                image_cache_path, self.dataset_root, len(self._states)
            )

    def _load_scalar_tensors(self) -> tuple[Tensor, Tensor]:
        paths = sorted((self.dataset_root / "data").glob("**/*.parquet"))
        if not paths:
            raise FileNotFoundError("No dataset parquet files")
        table = pa.concat_tables(
            [pq.read_table(path, columns=["observation.state", "action", "index"])
             for path in paths]
        )
        order = np.argsort(table["index"].to_numpy())
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)[order]
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("Global row indices must be contiguous from zero")
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
        return torch.from_numpy(states), torch.from_numpy(actions)

    def _build_window_index(self, selected: set[int]) -> list[tuple[int, int, int, int]]:
        metadata = self.source.meta.episodes
        known = {int(value) for value in metadata["episode_index"]}
        unknown = selected - known
        if unknown:
            raise ValueError(f"Unknown episode IDs: {sorted(unknown)}")
        windows: list[tuple[int, int, int, int]] = []
        for episode_id, start, end in zip(
            metadata["episode_index"], metadata["dataset_from_index"],
            metadata["dataset_to_index"], strict=True,
        ):
            episode_id, start, end = int(episode_id), int(start), int(end)
            if episode_id not in selected:
                continue
            for global_t in range(start, end - 1):
                windows.append((episode_id, global_t - start, global_t, end))
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    @staticmethod
    def _normalize_image(image: Tensor) -> Tensor:
        image = image.to(dtype=torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode_id, local_t, global_t, end = self._windows[index]
        image = (
            self._image_cache[global_t].clone()
            if self._image_cache is not None
            else self._normalize_image(self.source[global_t]["observation.image"])
        )
        position = self.normalization.normalize_position(self._states[global_t])
        valid_end = min(global_t + self.action_horizon, end - 1)
        count = valid_end - global_t
        actions = torch.zeros((self.action_horizon, 2), dtype=torch.float32)
        mask = torch.zeros(self.action_horizon, dtype=torch.bool)
        actions[:count] = self.normalization.normalize_action(
            self._actions[global_t:valid_end]
        )
        mask[:count] = True
        return {
            "observation_image": image,
            "agent_position": position,
            "action_chunk": actions,
            "action_valid_mask": mask,
            "episode_id": torch.tensor(episode_id, dtype=torch.int64),
            "start_step": torch.tensor(local_t, dtype=torch.int64),
        }

    def source_indices(self, index: int) -> tuple[int, int, int, int]:
        """Expose (episode, local step, global row, exclusive episode end)."""
        return self._windows[index]
