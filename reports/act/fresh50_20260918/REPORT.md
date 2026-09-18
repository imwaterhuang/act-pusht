# ACT fresh 50-scene checkpoint comparison

One training seed; the same 50 fresh random scenes for each checkpoint. Maximum 300 environment steps, execute 4 actions before reobserving, strict success: coverage > 0.95.

| Training step | Success | Success rate | Mean final coverage | Mean max coverage |
|---:|---:|---:|---:|---:|
| 30000 | 21/50 | 42.0% | 0.6726 | 0.7071 |
| 35000 | 14/50 | 28.0% | 0.6040 | 0.6565 |
| 40000 | 18/50 | 36.0% | 0.6223 | 0.6633 |

Scene generation seed: 2026091801. The manifest records disjoint seeds/states, source revision, checkpoint hashes and the scene hash.

Original development-selected best.pt and selection.json were preserved. This requested comparison does not establish a statistically reliable training-step advantage or robustness across training seeds.

## Additional coverage threshold

Counts use strictly > 0.85 and include >0.95 successes. Final and maximum coverage come from the original saved trajectories; no episodes were rerun.

| Step | Final >85% | Ever >85% | Ever >85%, final <=85% |
|---:|---:|---:|---:|
| 30000 | 31/50 | 33/50 | 2/50 |
| 35000 | 27/50 | 31/50 | 4/50 |
| 40000 | 29/50 | 32/50 | 3/50 |
