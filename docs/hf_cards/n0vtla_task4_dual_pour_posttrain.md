# N0-VTLA post-train, Dual-arm tube pour (`n0vtla_task4_dual_pour_posttrain/`)

All three tactile arms of the Dual-arm tube pour post-train live in this one folder, one sub-folder per arm. Same recipe for all three: init `n0-vtla-base_plus_stage1_human_14000`,
the same 92 source trajectories (batch 1 post-train split + batch 2, no train/val split), one dataset episode per source trajectory (whole-trajectory build), global batch 64 on 8 GPUs,
**40000 steps** (cosine LR decay over 40000 steps, run as resumed segments), checkpoint every 5000 steps. State, action, RGB, episodes and normalisation are identical; only the tactile channel differs.

| Folder | Tactile input | Dataset (VISION `data/n0vtla/`) |
|---|---|---|
| `task4_posttrain_ours_zerotac0_post92traj40k_vision8gpu/<step>/` | none (all-zero glove readings, tactile difference to the first frame is exactly zero) | `canonical_wetlab_task4_dual_post92traj_zerotac0` |
| `task4_posttrain_ours_predtacq_post92traj40k_vision8gpu/<step>/` | **predicted** from video (v2 DiT -> cell-to-taxel decoder, per-hand quantile map) | `canonical_wetlab_task4_dual_post92traj_predtacq` |
| `task4_posttrain_ours_realtac_post92traj40k_vision8gpu/<step>/` | **real** glove readings | `canonical_wetlab_task4_dual_post92traj_realtac` |

At run time the no-tactile policy should be given the same constant (zero) tactile input. `optimizer.pt` is not uploaded.
