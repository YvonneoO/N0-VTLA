# N0-VTLA post-train with PREDICTED tactile (`n0vtla_predtac_posttrain/`)

**These checkpoints were post-trained on tactile that was predicted from video, not on the glove.** The robot-right glove channel of the post-train
datasets was replaced by the output of a video-to-tactile predictor; everything else (state, action, RGB, episodes, normalisation statistics, recipe) is
identical to the real-tactile post-train runs of the TouchScale project (`n0vtla_wetlab_posttrain/`, `n0vtla_scaling_posttrain/`).

| Folder | Task | Dataset (VISION `data/n0vtla/`) |
|---|---|---|
| `task1_posttrain_ours_predtacq_vision8gpu/<step>/` | task1 (tube rack hole transfer) | `canonical_wetlab_task1_predtacq_train` (asset `task1_tuberack_train`) |
| `sv2_posttrain_ours_predtacq_vision8gpu/<step>/` | smoke_test_v2 (cap to tray) | `canonical_wetlab_smoketestv2_predtacq_train` (asset `wetlab_smoketestv2_train`) |

* Recipe: init `n0-vtla-base_plus_stage1_human_14000` (mid-train on human ITW TouchScale data), batch 64 on 8 GPUs, 20000 steps, saves every 5000, default LR schedule.
  `optimizer.pt` is not uploaded.
* Predictor: v2 DiT (WiLoR-feature conditioned flow matching) fine-tuned on robot glove data, checkpoint
  `qqyang/e2c-downstream-eval:robot_finetune/ckpt_694468/best_model.pth` (epoch 20, validation contact IoU 0.267). Inference is causal (window ending at each frame), 8 samples averaged.
* Predicted cells -> glove taxels: hard threshold 0.02, per-task ridge decoder with mass preservation, per-task quantile map (fitted on the real glove of the 53
  post-train training episodes of the task), then the task's own tactile norm. Decoders and maps: `qqyang/e2c-downstream-eval:robot_finetune/decoder/`.
  Figures (RGB | real | predicted through every layout): `robot_finetune/pred_viz/`.
* Quality of the predicted tactile on the 53 unseen post-train training episodes per task (policy units): total-force correlation 0.975 (task1) / 0.91 (sv2),
  force level 1.06 / 1.08, per-pad correlation 0.76 / 0.59. Full write-up: https://claude.ai/artifact/7MnqQvACYAcEDa8CkVLo3g
* Deployment is meant to use the REAL glove as tactile input; only the post-train signal is predicted.
