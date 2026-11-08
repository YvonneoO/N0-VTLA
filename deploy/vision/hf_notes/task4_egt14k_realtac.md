**EgoTouch mid-train init** -- `task4_posttrain_egt14k_realtac_post92traj40k_vision8gpu/<step>/`: same recipe and **real** glove tactile
(`canonical_wetlab_task4_dual_post92traj_realtac`, asset `task4_dual_pour_post92traj`, 40000 steps, cosine over 40000, every 5000),
but init = `n0-vtla-base` + Stage-1 mid-train on **EgoTouch** (`n0vtla_egotouch_midtrain/egt_20k/14000`, merged) instead of the
human ITW mid-train (`n0-vtla-base_plus_stage1_human_14000`). Compare with `task4_posttrain_ours_realtac_post92traj40k_vision8gpu`.
`optimizer.pt` is uploaded for the latest step of each training segment only.
