# Wet-Lab Post-Training Smoke

For download locations and the shortest launch instructions, see
[Robot Post-Training: Lab Setup](ROBOT_POSTTRAIN_QUICKSTART.md).

## Scope

This is an independent official-base robot post-training plumbing test, not a
human Stage-1 result or a deployable policy. It uses one successful
`cap_to_tray` recording from dataset revision
`ca16780a64856da90033d0270822a806ab3eb53c`, UUID
`094fa583-d87b-5c37-ad38-42d57fa97d48`. Execution is inside lab Docker `n0vtla`.
Code is edited locally, pushed to the fork, then pulled inside Docker.

## Alignment Evidence

Three different questions must not be conflated:

1. **Recorded timestamps and indices:** H5 camera indices match original CSV
   timestamps. Selected images precede each model tick by 14.78-15.32 ms (head)
   and 10.80-11.98 ms (right wrist). Raw pressure is sampled causally, with
   maximum recorded age 35.93 ms in the selected segment.
2. **Command freshness:** raw accepted arm commands and available hand command
   records are held causally on the unchanged H5 30 Hz grid. Reject targets
   older than 150 ms, pre-command rows and clutch-off rows. This produces one
   continuous run, original H5 rows 173-479 inclusive, 307 frames. Maximum arm
   command age is 101.85 ms and hand command age 136.35 ms. This age gate is a
   smoke data-preparation choice, not a validated robot control latency budget.
   No disjoint timestamps are concatenated. Episode-end repeat-last action
   padding remains the reference loader behavior.
3. **Physical cross-host synchronization:** NOT verified. The source has no
   trustworthy clock anchor or clap markers. An independent wrist optical-flow
   versus measured TCP-speed diagnostic gives whole-recording best lag
   -1266.7 ms (correlation 0.685 vs 0.419 at zero lag), first-half best lag
   +2400 ms, and second-half -1300 ms. This inconsistency prevents accepting a
   unique offset. No inferred correction was applied. Nominally causal indexing
   does not establish causality across unsynchronized physical clocks.

The adapter requires `--allow-unverified-sync-smoke` and records
`physical_sync_verified=false`. Do not use the smoke data for evidence of
temporal learning quality or for unsupervised robot deployment. Physical sync
needs independent markers/calibration or a reliably tracked shared event with
consistent estimates across segments; a QC PASS is insufficient.

## Action and Sensor Contract

- State is the last causally available issued arm and hand command, matching
  the dataset card's command-state contract. It is NOT measured proprioception.
  Online deployment would need the same last-command buffer.
- Canonical `[10:13]` is right EEF xyz in mm, `[13:19]` is the concatenated
  first two rotation-matrix columns, converted from axis-angle degrees.
  `[20:26]` retains the six Revo2 targets in original motor order and raw
  0..1000 units. Other dimensions are zero. This is versioned as
  `right_eef_mm_columns6d_10_19_revo2_raw6_20_26_v1`.
- Model shape stays 32-D. Reserved output channels gain dataset-specific motor
  meanings; they do not inherit pretrained six-finger semantics automatically.
- Stored actions are absolute commands. The reference loader subtracts current
  EEF state from all 50 action targets, including elementwise rot6d subtraction;
  it does NOT use Cosmos backward-framewise increments. Hand values stay absolute.
- `compute_canonical_norm.py --robot aloha` selects the exact existing two-EEF
  delta mask. This selects normalization semantics, not an ALOHA robot claim.
- Real sensors: static head RGB, right wrist RGB, right glove pressure. Missing
  RGB/tactile slots are omitted on disk and become zero/false model placeholders.
  The separate five-channel Revo2 fingertip stream is not silently substituted.
- Pressure uses the existing hand layout, replicated grayscale, black canvas,
  and `[-1,8]` transport mapping. A separate smoke-only robot train asset uses
  the same P5/P99.9 method, four sampled frames (seed 42), and scale floor.
  No human normalization file is changed. This one-recording fit is not a
  production normalization estimate: final statistics must be fitted across
  robot TRAIN recordings only and audited for clipping and calibration units.

## Verified Data Checks

- Four CPU unit tests pass: causal/stale boundaries, no gap compression,
  six-motor/rotation round trips, and invalid-command rejection.
- Actual reference-loader batch: actions `[64,50,32]`, state `[64,32]`, all finite.
  First sequential batch normalized action range is approximately [-1.096,1.145];
  quantile normalization intentionally does not clamp to [-1,1].
- Real right tactile/current and baseline masks are true for all 64 samples;
  missing left RGB/tactile masks are false. Post-training future tactile masks
  are false, as expected with the supervised predictor objective disabled.
- Raw command packing position round-trip error: 1.53e-5 mm; rotation error:
  4.75e-12 rad; hand command error: zero. All 307 current-state delta transforms
  match the normalization helper's action-chunk convention. The complete
  delta/quantile-normalization/inverse path has maximum error 7.63e-6.
- All 307 tactile video frames decode. Video is nonconstant: decoded range
  0..255, up to 1,939 pixels differ from its first frame. Visibility alone does
  not certify pressure calibration, synchronization, or absence of clipping.

## Reproduction

Run these commands only inside `n0vtla`, from `/DATA2/qianqian/N0-VTLA`.
The converter refuses to overwrite an existing output.

```bash
python scripts/wetlab_smoke_adapter.py \
  /DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test/094fa583-d87b-5c37-ad38-42d57fa97d48 \
  /DATA2/qianqian/n0vtla_robot_audit/canonical_smoke_v1 \
  --allow-unverified-sync-smoke

export VTLA_DATASET_PATH=/DATA2/qianqian/n0vtla_robot_audit/canonical_smoke_v1
export VTLA_ASSET_ID=wetlab_smoke_v1
export VTLA_PRETRAINED_CHECKPOINT=/DATA2/qianqian/N0-VTLA/checkpoints/n0-vtla-base
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/compute_canonical_norm.py --repo-id "$VTLA_DATASET_PATH" \
  --robot aloha --train-config-name vtla_tactile_posttrain --asset-id "$VTLA_ASSET_ID"
python -m unittest discover -s tests -p test_wetlab_smoke_adapter.py -v
python scripts/verify_wetlab_smoke.py batch \
  --output /DATA2/qianqian/n0vtla_robot_audit/batch_check_v2.json

export CUDA_VISIBLE_DEVICES=4 NPROC_PER_NODE=1
export EXP_NAME=wetlab_cap_to_tray_smoke_single_v1
CHECK_ONLY=1 bash train.sh
bash train.sh --num-train-steps=3 --log-interval=1
```

The global batch remains 64 (64 on one GPU), horizon 50, warmup 500, peak LR 2e-5,
decay schedule 20,000, tactile LR scale 0.1, eager attention and original action
loss. Only smoke duration and log frequency are overridden. No model config,
loss masking, architecture, or freeze setting is changed. The default task
prompt remains `Perform the task.`; this is a single-task smoke, not language
conditioning evaluation. The first attempt (`...smoke_v1`) was stopped before
an optimizer update after stalling in NCCL parameter broadcast. Four-GPU retry
`...smoke_v2` with P2P/IB disabled stalled at the same stage and was also stopped.
A separate small-tensor collective reproduced the communicator-init stall even
with cuMem host/device allocation disabled; no NCCL environment fix is claimed.
The collective's 45-second process-group timeout did not bound initialization;
use an external `timeout 90s` around such future diagnostics. All three stopped
attempts were cleaned up before the single-GPU run.

These are container-local communication diagnostics, not permission to change
host drivers, networking, or other users' jobs. The memory workaround investigated
is documented by [NVIDIA NCCL runtime troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html).

## Artifact Paths

All following paths are on lab inside Docker:

- Canonical data and audit: `/DATA2/qianqian/n0vtla_robot_audit/canonical_smoke_v1`
- Batch checks: `/DATA2/qianqian/n0vtla_robot_audit/batch_check_v1.json` and `batch_check_v2.json`
- Motion evidence: `/DATA2/qianqian/n0vtla_robot_audit/motion_diagnostic_v1.json`
- Training log: `/DATA2/qianqian/N0-VTLA/logs/wetlab_cap_to_tray_smoke_single_v1.log`
- Checkpoint: `/DATA2/qianqian/N0-VTLA/checkpoints/vtla_tactile_posttrain/wetlab_cap_to_tray_smoke_single_v1/2`
- Checkpoint audit: `/DATA2/qianqian/n0vtla_robot_audit/checkpoint_check_v1.json`

## Completed Smoke Result

The single-GPU run exited successfully on 2026-09-08 at 08:55:48 server log
time, completing all three optimizer steps. The training loop took about 114
seconds, including data-worker startup and checkpoint writing.

| Logged step (zero-based) | Loss | Gradient norm | Main LR |
|---|---|---|---|
| 0 | 0.7585564 | 7.42 | 3.9920e-8 |
| 1 | 0.8089033 | 7.35 | 7.9840e-8 |
| 2 | 0.6159542 | 5.19 | 1.1976e-7 |

No nonfinite-gradient skip was logged. Peak allocated GPU memory was 49.65 GiB,
peak reserved 52.64 GiB. The tiny learning rates are the unchanged 500-step
warmup, not evidence of a useful fitted policy. Three stochastic batches do not
establish a loss trend, convergence, dataset benefit, or real-world success.

Checkpoint deserialization passed: all 1,074 stored model tensors are finite;
optimizer state and metadata both report global step 2. Compared with official
base, action output projection, tactile projection, z projection and z gate
contain actual nonzero parameter changes. This validates optimizer updates,
including the tactile pathway, rather than only a forward pass.

**Existing checkpoint-save off-by-one:** the trainer increments `global_step`
before calling `save_checkpoint`, but its final-save condition compares against
`num_train_steps - 1`. Thus this three-step run writes `/2`, not `/3`; the third
update completed but was not persisted. No training code was changed to hide or
repair this during the smoke. Before a full run, separately fix/test final-step
checkpoint saving or choose an explicit save interval that covers the last step.

Status: data/codec/normalization/forward/backward/optimizer/save/read checks PASS;
multi-GPU NCCL initialization and physical cross-host synchronization remain
unresolved. No real-robot motion or deployment was performed.
