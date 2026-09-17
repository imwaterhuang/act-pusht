"""Evaluate a trained ACT checkpoint on frozen development or review scenes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from mini_wam.evaluation.act import evaluate_act_policy, load_act_checkpoint  # noqa: E402
from mini_wam.training.runtime import select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit", type=Path,
                        default=PROJECT_ROOT / "artifacts/act/data_audit.json")
    parser.add_argument("--scenes", type=Path,
                        default=PROJECT_ROOT / "splits/act_development_scenes.json")
    parser.add_argument("--split", choices=("development", "review"),
                        default="development")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--execute-steps", type=int, default=4)
    parser.add_argument("--save-videos", action="store_true")
    args = parser.parse_args()
    device = select_device(args.device)
    model, normalization, checkpoint = load_act_checkpoint(
        args.checkpoint, args.audit, device
    )
    result = evaluate_act_policy(
        model, normalization, args.scenes, args.output_dir,
        device=device, expected_split=args.split,
        max_steps=args.max_steps, execute_steps=args.execute_steps,
        save_videos=args.save_videos,
    )
    summary = result["summary"]
    print(f"checkpoint step: {checkpoint['step']}")
    print(f"strict successes: {summary['success_count']}/{summary['episode_count']}")
    print(f"mean final coverage: {summary['mean_final_coverage']:.6f}")


if __name__ == "__main__":
    main()
