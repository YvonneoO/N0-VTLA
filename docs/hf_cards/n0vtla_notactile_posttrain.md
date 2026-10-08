# N0-VTLA post-train with NO tactile signal (`n0vtla_notactile_posttrain/`)

Control for the predicted-tactile post-train (`n0vtla_predtac_posttrain/`). Same recipe, same init (`n0-vtla-base_plus_stage1_human_14000`, which was mid-trained
WITH tactile), same episodes, state, action, RGB, normalisation and 20000 steps; the only difference is the tactile channel of the dataset: **all-zero glove
readings**, i.e. the input of an idle glove ("no touch ever"; tactile difference to the first frame is exactly zero).

| Folder | Task | Dataset (VISION `data/n0vtla/`) |
|---|---|---|
| `task1_posttrain_ours_zerotac0_vision8gpu/<step>/` | task1 | `canonical_wetlab_task1_zerotac0_train` |
| `sv2_posttrain_ours_zerotac0_vision8gpu/<step>/` | smoke_test_v2 | `canonical_wetlab_smoketestv2_zerotac0_train` |

At run time this policy should be given the same constant (zero) tactile input, not the glove. `optimizer.pt` is not uploaded.
Compare with the real-tactile runs (`n0vtla_wetlab_posttrain/`, `n0vtla_scaling_posttrain/`) and the predicted-tactile runs (`n0vtla_predtac_posttrain/`).
