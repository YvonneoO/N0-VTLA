# Robot Post-Training: Lab Setup

## Status (updated 2026-09-09 — see [ROBOT_POSTTRAIN_OPEN_ISSUES.md](ROBOT_POSTTRAIN_OPEN_ISSUES.md) for full history)

The training pipeline is installed and smoke-tested in lab Docker `n0vtla`.
It initializes from the official N0-VTLA base, **not our human Stage-1 checkpoint**.

All 53 raw episodes are downloaded and converted into **three physically
separate canonical datasets** (train/val/holdout — see below for why they must
be separate directories, not one tagged directory), **superseding both
`canonical_smoke_v1` and the single combined `canonical_wetlab_v1` described
further below** (the former was built from a since-fixed dead tactile channel
and an over-strict camera-alignment check that discarded most episodes; the
latter silently trained on its val split too — see `ROBOT_POSTTRAIN_OPEN_ISSUES.md`
§3.8 — neither should be reused):

- **Use this one for training**: `/DATA2/qianqian/n0vtla_robot_audit/canonical_wetlab_v2_train`
  — 52 canonical episodes / 13,812 frames, from 42 mutually-exclusive source
  episodes. Sibling directories `canonical_wetlab_v2_val` (9 episodes / 1,820
  frames, 5 source) and `canonical_wetlab_v2_holdout` (7 episodes / 2,229
  frames, 6 source, for checkpoint-selection/deployment gating — never for
  training) hold the other two partitions. Built by
  `scripts/build_wetlab_canonical_dataset.py --which {train,val,holdout}` from
  a shared tactile normalization (`scripts/fit_wetlab_tactile_norm.py`, fit
  restricted to each train episode's own trial window) and the split manifest
  `scripts/wetlab_split_v1_tacwam_match.json` (train/val approximate tacWAM's
  own default 48/5 episode-random split, minus the 6 block-holdout episodes,
  for direct comparability; see the manifest's `train_tacwam_raw`/`val_tacwam_raw`
  for the un-adjusted lists tacWAM itself uses).
  **Why three directories and not one with a split tag**:
  `n0vtla/training/data_loader.py`'s loader has no concept of a per-episode
  split tag and doesn't read `meta/info.json`'s `splits` field either — it
  loads every episode under whatever root `VTLA_DATASET_PATH` points at,
  unconditionally. Physical separation is the only thing this loader respects.
- Train-only pose/action norm stats:
  `/DATA2/qianqian/N0-VTLA/assets/vtla_tactile_posttrain/wetlab_v2_train/norm_stats.json`
  (`compute_canonical_norm.py` against `canonical_wetlab_v2_train` directly —
  no `--train-only` flag needed now that the directory itself is train-only).
- Fixed since the original smoke: the delivered `right_hand_data.npz` is a dead
  channel on this rig (live signal is in `left_hand_data.npz`); the original
  camera causal-gap check silently discarded 31/53 episodes; and the training
  loader was silently training on val (and even holdout) episodes because
  nothing enforced the split at the physical-directory level (see
  `ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.1, §3.7, §3.8).

**A real full training run is in progress**: `wetlab_v2_train_full_run1`, 4-GPU
DDP (physical GPUs 6,0,1,2 — chosen because they were genuinely idle per
`nvidia-smi --query-compute-apps`, not just low `utilization.gpu`), 20,000
steps at the `vtla_tactile_posttrain` config's own defaults (batch 64 global /
16 per GPU, warmup 500, peak LR 2e-5, decay to 2e-6 over 20,000 steps,
`save_interval=5000`), against `canonical_wetlab_v2_train` from official base.
Multi-GPU DDP was previously believed broken on this host (see
`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.6) — retested and it works; the earlier
stalls were caused by GPU contention with other users' jobs, not a code or
container-config bug. Preceded by a 300-step DDP validation run on the same
dataset (steady-state ~5s/step on 4 GPUs vs ~9.5s/step on 1) that confirmed
training steps, gradient sync, and a full checkpoint save (step 100:
`model.safetensors` + `optimizer.pt`, training continued normally afterward)
all work correctly — that validation run and all older superseded artifacts
(`canonical_smoke_v1`, `canonical_wetlab_v1`, their checkpoints and norm
stats, `wetlab_full_v1_speedtest`) have been deleted from lab (~44 GB freed).

Still not done: robot controller integration and any real-world deployment.
Physical cross-host sync remains permanently unverified for this dataset (see
`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.3) — this training run does not resolve
that, and any resulting checkpoint should still be treated as
`physical_sync_verified: false` regardless of how well it trains.

## Data and Files

Source: [SingleBicycle/tacwam-wetlab-tasks](https://huggingface.co/datasets/SingleBicycle/tacwam-wetlab-tasks),
pinned revision `ca16780a64856da90033d0270822a806ab3eb53c`. All 53 episodes are
now downloaded to `/DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test`
(977 files, ~13 GB): H5, head/right-wrist RGB and timestamps, **both**
`left_hand_data.npz` and `right_hand_data.npz` (needed to confirm which one is
live — see the swap above), robot JSONL, and metadata. Depth video was not
downloaded (not needed by any current check).

The rest of this document describes the original **single-episode smoke**
(`canonical_smoke_v1`) that validated the training loop end-to-end. It is kept
for the reproduction commands and the multi-GPU/checkpoint-saving findings,
which are still accurate — just not for the data-pipeline details above, which
`ROBOT_POSTTRAIN_OPEN_ISSUES.md` supersedes.

All paths below are on lab, visible inside `n0vtla`:

| Item | Absolute path |
|---|---|
| Repository | `/DATA2/qianqian/N0-VTLA` |
| Download root | `/DATA2/qianqian/n0vtla_robot_audit/cap_to_tray` |
| Raw episode | `/DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test/094fa583-d87b-5c37-ad38-42d57fa97d48` |
| Prepared dataset | `/DATA2/qianqian/n0vtla_robot_audit/canonical_smoke_v1` |
| Official base checkpoint | `/DATA2/qianqian/N0-VTLA/checkpoints/n0-vtla-base` |
| State/action statistics | `/DATA2/qianqian/N0-VTLA/assets/vtla_tactile_posttrain/wetlab_smoke_v1/norm_stats.json` |
| Pressure statistics | `/DATA2/qianqian/n0vtla_robot_audit/canonical_smoke_v1/meta/tactile_normalization.json` |
| Smoke log | `/DATA2/qianqian/N0-VTLA/logs/wetlab_cap_to_tray_smoke_single_v1.log` |
| Saved smoke checkpoint | `/DATA2/qianqian/N0-VTLA/checkpoints/vtla_tactile_posttrain/wetlab_cap_to_tray_smoke_single_v1/2` |

The local code/docs checkout is
`/Users/yvonney/Library/CloudStorage/OneDrive-Personal/@TAMU/N0-VTLA`.
Edit locally, push to `YvonneoO/N0-VTLA`, then pull inside Docker. Do not edit
server code independently. Credentials stay outside Git.

## Data Contract and Changes

```text
Raw H5 + robot JSONL + RGB + tactile NPZ
  -> wetlab_smoke_adapter.py
  -> 307 continuous frames at 30 Hz: canonical Parquet + 224x224 MP4
  -> compute_canonical_norm.py
  -> official base + vtla_tactile_posttrain + action loss
```

- **Inputs:** static head RGB, right wrist RGB, right tactile current/baseline,
  and last-commanded arm/hand state. State is NOT measured proprioception.
- **Targets:** 50-frame command chunks. Keep right EEF xyz in mm at `[10:13]`,
  column-based rot6d at `[13:19]`, and six Revo2 motor targets at `[20:26]`.
  Other channels are zero; missing sensor slots receive false masks.
- **Action transform:** subtract the current EEF state from every chunk target;
  hand targets remain absolute. Use `--robot aloha` in the normalization helper
  to select the matching two-EEF delta mask, not to identify the hardware.
- **Tactile:** reuse Stage-1's pressure-only hand layout, black background and
  grayscale encoding. Refit separate smoke-only robot statistics with the same
  tacWAM P5/P99.9 method; do not reuse human scale values or modify human assets.
- **Difference from tacWAM:** its recorded Cosmos experiment uses measured
  streams and backward-framewise increments at 10 Hz. Those targets/statistics
  are not interchangeable with this command-based 30 Hz N0-VTLA adapter.
- No model architecture or training-recipe changes were made. Added code handles
  robot conversion, format/round-trip checks, and timing diagnostics.

## Run the Existing Smoke Dataset

Enter the container from the local terminal:

```bash
ssh -t lab 'docker exec -it -w /DATA2/qianqian/N0-VTLA n0vtla bash'
```

Run inside Docker. Check that GPU 4 is free before selecting it; it was the
96 GB GPU used for the completed smoke. Use a new experiment name, without overwrite.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=4 NPROC_PER_NODE=1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VTLA_DATASET_PATH=/DATA2/qianqian/n0vtla_robot_audit/canonical_smoke_v1
export VTLA_ASSET_ID=wetlab_smoke_v1
export VTLA_PRETRAINED_CHECKPOINT=/DATA2/qianqian/N0-VTLA/checkpoints/n0-vtla-base
export CONFIG_NAME=vtla_tactile_posttrain
export EXP_NAME="wetlab_cap_smoke_$(date +%Y%m%d_%H%M%S)"
CHECK_ONLY=1 bash train.sh
bash train.sh --num-train-steps=3 --log-interval=1
```

Unchanged recipe: batch 64, horizon 50, warmup 500, peak LR `2e-5`, decay to
`2e-6` over 20,000 steps, tactile LR multiplier `0.1`, gradient clip `1.0`,
eager attention, and no supervised predictor loss. Only smoke duration/logging
are overridden. The single-task smoke retains the default `Perform the task.` prompt.

## Results and Remaining Work

Smoke loss: `0.7586 / 0.8089 / 0.6160`; gradients finite; peak allocated GPU
memory 49.65 GiB. All 1,074 saved model tensors are finite, with verified action
and tactile parameter updates. This is pipeline validation, not policy success.

- **Synchronization:** recorded indices and causal sampling pass, but physical
  cross-host alignment remains unverified. No inferred offset was applied.
- **Pressure:** current statistics are a one-recording smoke fit. Verify sensor
  units/calibration and fit fixed statistics across robot TRAIN data before full training.
- **Runtime:** multi-GPU NCCL initialization stalls; the completed run used one GPU.
- **Saving:** the original trainer saves step 2 for this three-step run due to
  a final-save off-by-one. The third update was not persisted.
- **Deployment:** still needs matching online observations/command-state buffer,
  action decoding, controller integration, safety limits, and supervised trials.

See [the detailed smoke report](WETLAB_POSTTRAIN_SMOKE.md) for conversion commands,
audit results, and diagnostics; [the study overview](HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md)
separates the human-pretraining study from the official-base robot experiment.
