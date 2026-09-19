## n0vtla_pretrain_scaling_study/ — Stage-1 human-ITW data-scaling study

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
`stage1_recon` ≈ 0.0036–0.0040 for all. Differences (≤0.005 nce) are non-monotonic and within noise: this
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

**v3 (pending).** Task1 post-train-data retrieval; results appended when the jobs finish.
