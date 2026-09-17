"""ACT training command with clean stop, resume and optional artifact mirror."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data/.cache/hf_datasets"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mirror-dir", type=Path)
    args = parser.parse_args()
    from mini_wam.training.act import train_act
    run_path = train_act(
        args.config,
        resume_path=args.resume,
        run_dir=args.run_dir,
        stop_after_step=args.stop_after_step,
        device_name=args.device,
        num_workers=args.num_workers,
        mirror_dir=args.mirror_dir,
    )
    print(f"ACT training artifacts: {run_path}")


if __name__ == "__main__":
    main()
