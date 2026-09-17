# ACT Push-T on Colab

The September 17 run is active on NVIDIA L4. Source revision: `53dddff105b664612bedcf150cbc1f1fd2542bb6`.

- All 206 demonstrations, 25,444 valid windows; batch 64, 2 loader workers.
- 40,000 updates, learning rate 1e-4, warmup 500, KL weight 10.
- Save and mirror every 1,000 steps; select on 50 frozen random development scenes every 5,000 steps.
- Strict successes first, final coverage second, earlier step third.

## Authentication

Colab Secrets contains `GH_TOKEN`: repository-scoped Contents/Metadata read-only access to `imwaterhuang/act-pusht`, expiring October 17, 2026. Read it with `google.colab.userdata.get`, put it in the process environment, and use `gh auth setup-git`. Never embed the token in notebook source, outputs, remotes, or repository files.

Mount Google Drive with `drive.mount('/content/drive')`. Google may request authorization again after a new runtime; no permanent prompt-free mount is promised.

## Verified gates

`validate_act_colab.py` measured batch sizes 32/64 with 0/2/4 workers, checked consumed tensors, compared full continuous/resumed state after recovering from Drive, and ran a 64-window reconstruction diagnostic. `check_act_rollout.py` checked pretrained initialization, actual policy rollouts, best-checkpoint selection and mirroring. See [validation evidence](reports/act/colab_validation_20260917.json).

Full validation records: `MyDrive/act-pusht-runs/validation-20260917-154249/`.
Formal run: `MyDrive/act-pusht-runs/act-seed0-20260917-l4/`.

## Recovery

Open [the recovery notebook](notebooks/act_colab_resume.ipynb) in Colab after the original runtime stops. Select L4 (or T4 if unavailable), authorize the notebook to read `GH_TOKEN`, mount Drive and run the cells. It restores the complete mirrored run to local disk, checks out the original code revision, and resumes from the latest checkpoint. It refuses a duplicate running process in the same runtime or an already-completed run.

Do not execute two runtimes against the same Drive mirror. The original training notebook is [here](https://colab.research.google.com/drive/1YAk4p-ptP7bnXtL-MY7y9gCUndLPyXRh).

A configured 40,000-step target is not a completed run. Completion requires `completed.json`, actual step 40,000, and matching mirrored checkpoint state.
