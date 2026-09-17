"""Freeze the full 206-episode ACT manifest and normalization statistics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data/.cache/hf_datasets"))

from mini_wam.data.act_audit import save_act_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "data/lerobot/pusht_image")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "splits/act_all_episodes.json")
    parser.add_argument("--audit", type=Path,
                        default=PROJECT_ROOT / "artifacts/act/data_audit.json")
    args = parser.parse_args()
    manifest, audit = save_act_audit(args.dataset_root, args.manifest, args.audit)
    if manifest["episode_count"] != 206 or audit["valid_window_count"] != 25444:
        raise RuntimeError("Current Push-T source must contain 206 episodes and 25,444 valid windows")
    print(f"ACT: {manifest['episode_count']} episodes, {audit['valid_window_count']} valid windows")
    print(f"Manifest: {args.manifest}")
    print(f"Audit: {args.audit}")


if __name__ == "__main__":
    main()
