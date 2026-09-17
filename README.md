# ACT Push-T

Standalone training release of the user-written ACT (Action Chunking with Transformers) policy.

- Train on all 206 demonstrations: 25,650 frames and 25,444 valid action windows.
- Current image and agent position predict 16 absolute-target actions; execute 4 before observing again.
- Choose checkpoints on 50 frozen random development scenes: strict successes, then mean final coverage, then earlier step.
- Full optimizer, scheduler, sampler and random state recovery; optional Google Drive mirror.

The ACT model and training implementation are preserved from the source workspace. Shared `mini_wam` modules include legacy helpers required by package imports; this release runs only ACT. Dataset, weights, credentials and legacy experiment results are excluded.

## Colab

See [COLAB.md](COLAB.md). Install `requirements-colab.txt` and the package with `pip install -e . --no-deps`, download the dataset with `python scripts/download_dataset.py`, then run `configs/act_smoke.yaml`. L4 profiling and recovery gates passed; formal seed-0 training started with batch 64, 2 workers and 40,000 updates. See reports/act/colab_validation_20260917.json.

See [SPEC.md](SPEC.md), [PROGRESS.md](PROGRESS.md), and [EVALUATION.md](EVALUATION.md) for the experiment protocol. Historical links refer to the original Mini-WAM workspace, not results of this ACT release.
