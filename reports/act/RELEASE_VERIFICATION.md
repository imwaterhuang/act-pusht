# Release verification · 2026-09-18

Validation ran against the independent `act-pusht` publish directory, using Python 3.12 on macOS. The original training ran on Python 3.13.15 and NVIDIA L4; these environments are recorded separately.

- Full release test suite with local dataset and the real 30,000-step checkpoint: **35 passed, 3 skipped**. The skips require historical action-only / legacy / overfit checkpoints that are outside this ACT release.
- Without optional local data and weights: 32 passed, 6 skipped. Skips are not counted as successful model validation.
- Added actual rollout boundary tests: coverage exactly 0.87 does not succeed; 0.871 stops the 87% rollout, while the 95% run continues until 0.96. Invalid thresholds are rejected before loading scenes.
- Actual-checkpoint test verifies the web adapter matches formal inference and a short rollout; changing the unused historical frame does not affect ACT output.
- Live demo returned an actual model frame through its WebSocket connection; pause took effect. The HTML (HyperText Markup Language) page returned HTTP (Hypertext Transfer Protocol) status 200.
- All five videos decoded with the expected frame count. Entire source trajectories were replayed with absolute coverage error below 1e-8. Final-frame contact sheet and training plots were visually inspected.
- Original 40,000-row training log has steps 1 through 40,000 exactly once; completion marker reports `completed=true, step=40000`.
- All 150 original review trajectories were checked against their saved maxima and final coverage before threshold rescoring.
- README relative media links and document links were checked. GitHub's Markdown renderer preserves the HTML preview table and filters ordinary video tags, hence the linked animated previews plus local video gallery.
- No credential-like tokens were found in the release text scan. Data directories, training outputs and model binaries are excluded from Git; inference weights are distributed as a Release asset.

The shared dataset-helper test originally assumed a legacy audit file was always present. The standalone release does not include that unrelated file; the test now checks the documented raw-data statistics when no legacy audit is present. ACT inference continues to use checkpoint normalization, verified independently.

These checks do not constitute a multi-seed benchmark or a guarantee of bitwise-identical inference across devices.
