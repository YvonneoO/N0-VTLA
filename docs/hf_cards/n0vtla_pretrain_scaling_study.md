## n0vtla_pretrain_scaling_study/ — mid-train (formerly Stage-1) human-ITW data-scaling study

> Naming: the former "Stage-1 / pretrain" step is now called **mid-train**. Folder names, `stage1_*` metric names and the
> config name `vtla_stage1_predictor_pretrain` are unchanged so existing paths keep working. The study has three parts:
> mid-train (this folder), post-train and offline eval (`n0vtla_scaling_posttrain/`, below).

N0-VTLA **Stage-1 predictor-grounding pretraining** (paper Sec 4.2) on human in-the-wild (ITW)
tactile data, trained on nested fractions {20, 40, 60, 80, 100}% of the corpus, then scored on
one **shared held-out episode set** none of them trained on. Everything is warm-started from
`n0-vtla-base`; only the tactile encoder projection, tactile predictor and a small recon head
(~123M params) are trained (base policy frozen), 14000 steps, global batch 64, 8xH200.

### Data split (why the comparison is fair)
- One seeded shuffle (seed 0) of the ITW raw corpus (39,426 episodes). 20/40/60/80% are prefixes
  of that same list, so splits are strictly nested (20 ⊂ 40 ⊂ 60 ⊂ 80).
- `held_out` = the complement of the 80% prefix: by construction in **no** training split.
  After a file-presence QC: 20%=7,885 · 40%=15,770 · 60%=23,656 · 80%=30,791 · held_out=7,731 episodes.
- 100% is a **reference point from a separate, earlier run** (`n0-vtla_ts_pretrain/14000`) trained on
  a different account's own pull of the corpus (mostly the same dates, not identical). Not drawn from
  the same shuffle; the held-out episodes were almost certainly unseen by it, but read the 100% point
  as "comparable distribution", not "same pool".
- Each run's own training loss is NOT a scaling metric: all fractions train the same 14000 steps, so a
  small pool is recycled more often and its training loss is biased low by memorization.

### Checkpoint locations
| Fraction | HF (Stage-1 trainable-only delta) | VISION delta (`.../yqq/N0-VTLA_scaling_extract/checkpoints/vtla_stage1_predictor_pretrain/`) | VISION merged base+delta (post-train ready, `.../yqq/data/n0vtla/`) |
|---|---|---|---|
| 20% | `n0vtla_pretrain_scaling_study/20pct/14000/` | `stage1_online_20pct/14000` | `n0-vtla-base_plus_stage1_scaling_20pct_14000` |
| 40% | `n0vtla_pretrain_scaling_study/40pct/14000/` | `stage1_online_40pct/14000` | `n0-vtla-base_plus_stage1_scaling_40pct_14000` |
| 60% | `n0vtla_pretrain_scaling_study/60pct/14000/` | `stage1_online_60pct/14000` | `n0-vtla-base_plus_stage1_scaling_60pct_14000` |
| 80% | `n0vtla_pretrain_scaling_study/80pct/14000/` | `stage1_online_80pct/14000` | `n0-vtla-base_plus_stage1_scaling_80pct_14000` |
| 100% (ref) | `n0-vtla_ts_pretrain/14000/` | `stage1_online_100pct_phailab/14000` | `n0-vtla-base_plus_stage1_human_14000` |

Delta = `model.safetensors` + `optimizer.pt` + `metadata.pt` (trainable params only). Post-training
needs a **complete** model, so each delta is merged onto `n0-vtla-base` into one 8.25 GB
`model.safetensors` (not on HF; `VTLA_PRETRAINED_CHECKPOINT` points at the merged dir).

### held_out_eval_results/ — one sub-directory per evaluation protocol
Every sub-directory holds `<label>.json` (raw per-checkpoint report, label = 20pct/40pct/…),
`summary.csv` (one row per checkpoint) and `logs/eval_<slurm-job>.log` (per-batch progress + final report).
All protocols score the same held_out set with the same 200 shuffled batches (batch 64, ~11.8k valid
frame samples; the loader yields one sample per FRAME, so a full pass is ~74k batches — infeasible —
and a fixed random subsample is used). Lower is better for losses.

| Directory | What it looks at |
|---|---|
| `v1_batchnce_recon/` | The Stage-1 training objective itself, evaluated on held-out. **`mean_stage1_nce`**: symmetric InfoNCE (temperature 1) between the predicted future-tactile latent z (mean-pooled, L2-normalised) and the real future latent z*, negatives = the other ~59 valid samples in the same batch. Chance = ln(59) ≈ 4.08. **`mean_stage1_recon`**: L1 between the recon head's 8x8 map (from z) and the 8x8-pooled future tactile field. **`mean_stage1_total`** = nce + 0.5·recon (the training loss). |
| `v2_poolretr_reconbase/` | Same forward pass, more sensitive readouts. **`retrieval`**: each predicted latent (query) is ranked against ALL ~11.8k real future latents in the pool (not just its batch) → top-1/5/10/100 accuracy, MRR, median rank (both i2t and t2i), `pool_nce`, `mean_pos_cos` vs `mean_neg_cos` (alignment margin), `chance_top1`=1/pool. **`recon_baselines`**: the recon head's L1 next to trivial predictors (all-zero, scalar mean, pool-mean field), overall and on contact cells (target above the 90th percentile), plus `skill_vs_pool_mean_field` = 1 − L1_model / L1_baseline (>0 means better than predicting the average map). |
| `v3_task1ptdata_retr/` | The same retrieval readouts on **post-train (Task1 tuberack, robot wetlab) data**, unseen by every Stage-1 checkpoint (they only trained on human ITW): a cross-embodiment transfer test in the post-train domain. Checkpoints are the **merged** base+Stage-1 models plus `base` = `n0-vtla-base` with no Stage-1 (reference). Splits: `train` (55 episodes, every 5th frame, capped at 6000 samples) and `val` (13 episodes, every 2nd frame), scored per split and **pooled** (train+val in one joint pool). `summary.csv/json` hold i2t/t2i top-1/5/10/100, MRR, pool_nce and mean positive cosine with episode-clustered bootstrap 95% CIs and paired differences vs `base`; `embeddings/` has the raw normalised (z, z*) per checkpoint/split for re-pooling. |

### Results and interpretation
**v1 (done).** Held-out `stage1_nce` (lower is better): 20%=3.8001 · 40%=3.8021 · 60%=3.8050 · 100%(ref)=3.7995;
`stage1_recon` ≈ 0.0034–0.0040 for all. Differences (≤0.005 nce) are non-monotonic and within noise: this
protocol shows **no data-scaling trend**. Adding 80% (`stage1_nce`=3.8014) does not change this. Contributing factors: temperature 1 with cosine logits compresses
InfoNCE's dynamic range (achievable floor ≈ 3.1 vs chance 4.08, and all models sit at ~3.80, i.e. close to
chance), and the 8x8 recon L1 is tiny/saturated (contributes ~0.002 of the total). Training curves plateau
at the same ~3.79–3.80. The v2 protocol was added to test whether a more sensitive readout separates them.

**v2 (done, all five points).** Pool retrieval on the 11,772-frame held-out pool (chance top-1 = 0.0085%):
i2t top-1 = 1.42 / 1.35 / 1.17 / 1.26 / 1.25 % for 20 / 40 / 60 / 80 / 100%; i2t top-10 = 7.8 / 7.3 / 6.9 / 7.2 / 7.1 %;
t2i top-1 = 3.54 / 3.19 / 3.20 / 2.78 / 2.57 %; pos−neg cosine margin ≈ 0.30 for all. The predictor clearly learns
(top-1 ≈ 150x chance) but shows **no data-scaling gain** — 20% is nominally best; the gaps are within ~2 sigma with
non-independent adjacent frames, so read it as a plateau, not as "more data hurts". Recon: the head's L1
(0.0034–0.0040) is ~7x **worse** than predicting all zeros (0.00052) overall and on contact cells
(0.0047–0.0051 vs 0.0025); the target field is near zero (mean ≈ −1e-5), so v1's small recon value was not
evidence of good reconstruction.

**v3 (done).** Task1 post-train-data retrieval (i2t; chance top-10 = 0.18% train / 0.33% val / 0.12% pooled;
CIs = episode bootstrap, in `summary.csv/json`):

| ckpt | train top-10 | train MRR | val top-10 | pooled top-10 | pooled pool_nce |
|---|---|---|---|---|---|
| base (no Stage-1) | 0.18% | 0.0016 | 0.62% | 0.14% | 9.031 |
| 20% | 2.87% | 0.0083 | 2.93% | 2.34% | 8.890 |
| 40% | 3.11% | 0.0090 | 2.80% | 2.63% | 8.889 |
| 60% | 3.18% | 0.0108 | 3.13% | 2.58% | 8.884 |
| 80% | 3.24% | 0.0094 | 3.13% | 2.69% | 8.890 |
| 100% (ref) | 3.31% | 0.0094 | 4.17% | 2.49% | 8.883 |

1. **Stage-1 clearly helps in the robot domain.** `base` retrieval is at chance (train top-10 0.18% = chance;
   pool_nce 8.584 ≈ ln 5469 = 8.607). Any Stage-1 checkpoint lifts train top-10 ~16–18x and MRR 5–7x, and lowers
   pool_nce by 0.145; paired 95% CIs exclude 0 (val and pooled agree in direction).
2. **Extra data adds little, and unstably.** Train top-10 rises monotonically with data (+0.24 / +0.31 / +0.37 /
   +0.44 pp vs 20%; CI excludes 0 for 40/80/100%), but the absolute gain is ~0.4 pp, per-checkpoint CIs overlap
   heavily, val (13 episodes) shows no significant differences and pooled top-10 is non-monotonic. Read it as a weak
   trend on the train split, not evidence of gains beyond ~20%.
3. **Absolute transfer is weak** (top-1 0.03–0.4%, positive cosine ~0.15 vs ~0.32 in-domain human held-out).

**Overall.** On human held-out data no metric shows a data-scaling benefit; on robot data Stage-1 itself matters a
lot but 20→100% adds only a weak, unstable difference. These are proxies; post-train and its offline evaluation follow
below, and real-robot rollouts (hardware team) decide.

---

## n0vtla_scaling_posttrain/ — post-train of the mid-train checkpoints + offline eval

### Post-train runs
Each merged mid-train checkpoint (20/40/60/80%) was post-trained on two robot tasks, **smoke_test_v2** (cap-to-tray, 59
train episodes, hand command is essentially two fixed poses: open / grasp) and **Task1** (tube-rack hole transfer, 55
train episodes): 8 runs, global batch 64, 8xH200, 10,000 steps (config-default lr schedule, peak 2e-5, warmup 500,
cosine decay over 20,000 steps, so the 10k checkpoint is at the same schedule state as the 10k checkpoint of the earlier
20k-step baseline/ours runs), ~3 h each. Real-robot rollouts are evaluated by the hardware team.

`<exp>/<step>/` with `<exp>` = `{sv2,task1}_posttrain_s1_{20,40,60,80}pct_10k` and steps 5000 / 9999 / 10000, each with
`model.safetensors`, `metadata.pt`, `assets/` (no `optimizer.pt`; that stays on VISION).

### Reference points (0% / 100%)
- Task1: `n0vtla_wetlab_posttrain/post_train_task1_vision8gpu/{baseline,ours}/10000` (official base / Stage-1-human-14000 base, 20k-step runs, step 10000).
- smoke_test_v2 100%: `n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_10000` (lab, 6 GPUs).
- smoke_test_v2 0%: `n0vtla_wetlab_posttrain/checkpoint_20000` = official base post-trained on the OLDER smoke_test data
  (asset `wetlab_v2_train`, own norm stats). **Not a like-for-like reference.**

### offline_eval/ — one sub-directory per protocol (each: per-checkpoint `<task>_<pct>.json`, `summary.csv`, `logs/`)
Only the final step-10000 checkpoint is evaluated; smoke_test_v2 checkpoints on `canonical_wetlab_smoketestv2_dev` (7
episodes), Task1 checkpoints on `canonical_wetlab_task1_val` (13 episodes), each with its own training normalization.
Cells: {0,20,40,60,80,100}% x {smoke_test_v2, Task1}.

| Directory | What it looks at |
|---|---|
| `heldout_action_loss/` | The flow-matching action loss the policy is trained with (denoising-field MSE in normalized space), computed with no gradient on held-out episodes: offline imitation fit, not closed-loop success. Every 5th frame, mean of 4 noise/time draws per frame with row-seeded draws shared across checkpoints. `mean_action_loss_micro/macro` + `episode_se`; `by_group` splits it into xyz / rot6d / hand (6 Revo2 motor commands); `by_horizon` gives the loss per step of the 50-step chunk. |
| `open_loop_action_error/` | Samples a 50-step chunk (10 denoising steps) per scored frame and compares it with the demonstration in physical units: xyz L2 error in mm, hand command MAE in raw counts (0-1000), rot6d MAE. Every 10th frame, 2 samples per frame. Baselines: predict zero motion (xyz/rot6d) and the training-set mean hand command. `metrics.<sample|mean|zero>_<xyz_mm|hand_mae|rot6d_mae>` with `mean`, `episode_se`, `by_horizon`. |

### Results and interpretation
Flow loss (total, mean ± episode SE): smoke_test_v2 20/40/60/80% = 0.0141 / 0.0137 / 0.0139 / 0.0138 (±0.0021); Task1
0/20/40/60/80/100% = 0.0556 / 0.0548 / 0.0554 / 0.0557 / 0.0557 / 0.0556 (±0.0055-0.0060).
Sampled xyz error (mm): smoke_test_v2 20/40/60/80% = 11.63 / 11.62 / 11.54 / 11.40 (±0.4; zero-motion baseline 36.25);
Task1 0/20/40/60/80/100% = 15.76 / 15.28 / 14.99 / 15.18 / 15.05 / 15.39 (±1.6-1.9; zero-motion 21.22). Hand MAE is
flat at 13.1-13.3 (smoke_test_v2) and 16.4-17.2 (Task1) counts vs 228 / 280 for the mean-command baseline.

1. **No measurable effect of mid-train data quantity** (20-80%) on either offline metric: differences are within one
   standard error; paired per-episode differences vs 20% are at most ~3% of the loss and not monotonic.
2. **Not even mid-train vs none is distinguishable on Task1** (0% vs 100%: loss 0.0556 vs 0.0556, xyz 15.76 vs 15.39 mm),
   although the same checkpoints differ hugely in tactile-latent retrieval (v3 above): post-training washes the
   difference out offline.
3. The policies are genuinely useful: sampled xyz error is ~30% (Task1) / ~68% (smoke_test_v2) below the no-motion baseline.
4. smoke_test_v2 0% and 100% are not comparable in total loss / rot6d (loss 0.849 / 0.430 and rot6d open-loop error
   0.335 vs 0.0005-0.0024 for 20-80%; the two rot6d errors are almost identical, which points to a rotation-representation
   mismatch between those earlier checkpoints and the current pipeline - not verified). Their xyz / hand terms are fine
   for the 100% (11.71 mm, 13.0 counts). The 0% (older data) fails on smoke_test_v2 (38.8 mm, worse than no motion).

Limits: 7 / 13 held-out episodes, one seed per fraction, final checkpoint only, and offline fit is not closed-loop success.
