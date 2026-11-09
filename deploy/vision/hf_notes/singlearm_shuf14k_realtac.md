**Shuffled-pair human mid-train init, single-arm tasks** -- `<task>_posttrain_shuf14k_realtac_vision8gpu/20000/` for st (original smoke_test, cap_to_tray),
task1 (tube rack), task2 (soft/hard sorting), task3 (whiteboard erasing): same recipe and **real** glove tactile as the
single-arm ours runs in `n0vtla_wetlab_posttrain/post_train_<task>_vision8gpu/ours` and the EgoTouch-init runs in
`n0vtla_singlearm_egt14k_posttrain/`: datasets `canonical_wetlab_<task>_train` (st: `canonical_wetlab_smoketest_train`), bs64 on
8 GPUs, 20000 steps, cosine over 20000. Init = `n0-vtla-base` + Stage-1 mid-train on the old-100% human ITW data with
**shuffled vision-tactile pairs** (ablation; `n0vtla_touchscale_midtrain/ts_old100_shuf_14k/14000`, merged) instead of the
correctly paired human ITW mid-train (`n0-vtla-base_plus_stage1_human_14000`). Only the final step (20000, with
`optimizer.pt`) is uploaded.
Runs: `st_posttrain_shuf14k_realtac_vision8gpu`, `task1_posttrain_shuf14k_realtac_vision8gpu`,
`task2_posttrain_shuf14k_realtac_vision8gpu`, `task3_posttrain_shuf14k_realtac_vision8gpu`.
