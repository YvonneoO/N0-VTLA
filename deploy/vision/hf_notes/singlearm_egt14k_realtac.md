**EgoTouch mid-train init, single-arm tasks** -- `<task>_posttrain_egt14k_realtac_vision8gpu/<step>/` for st (original smoke_test, cap_to_tray), task1 (tube rack),
task2 (soft/hard sorting), task3 (whiteboard erasing): same recipe and **real** glove tactile as the
single-arm ours runs in `n0vtla_wetlab_posttrain/post_train_<task>_vision8gpu/ours`:
datasets `canonical_wetlab_<task>_train` (st: `canonical_wetlab_smoketest_train`, built on VISION the same way), bs64 on 8 GPUs, 20000 steps, cosine over 20000. Init = `n0-vtla-base` + Stage-1 mid-train
on **EgoTouch** (`n0vtla_egotouch_midtrain/egt_20k/14000`, merged) instead of the human ITW mid-train
(`n0-vtla-base_plus_stage1_human_14000`). Steps kept: every 2000 and 19999; `optimizer.pt` only for the latest step of each
training segment.
Runs: `st_posttrain_egt14k_realtac_vision8gpu`, `task1_posttrain_egt14k_realtac_vision8gpu`,
`task2_posttrain_egt14k_realtac_vision8gpu`, `task3_posttrain_egt14k_realtac_vision8gpu`.
