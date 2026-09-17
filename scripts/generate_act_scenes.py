"""Freeze ACT development and fresh review Push-T scenes before evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from mini_wam.evaluation.act import generate_act_scene_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "review"), required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--development-count", type=int, default=50)
    parser.add_argument("--review-count", type=int, default=100)
    args = parser.parse_args()
    if args.split == "review":
        excluded = (
            PROJECT_ROOT / "splits/dev_scenes.json",
            PROJECT_ROOT / "splits/act_development_scenes.json",
        )
    else:
        excluded = (PROJECT_ROOT / "splits/dev_scenes.json",)
    payload = generate_act_scene_split(
        PROJECT_ROOT / f"splits/act_{args.split}_scenes.json",
        split=args.split,
        generation_seed=args.seed,
        count=args.development_count if args.split == "development" else args.review_count,
        excluded_paths=excluded,
    )
    print(f"ACT {args.split} scenes: {payload['count']}")


if __name__ == "__main__":
    main()
