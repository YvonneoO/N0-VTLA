# N0-VTLA training image (Stage-1 + Stage-2 + post-train)

Packages the `Stage-1 → Stage-2 → post-train → offline eval` chain. Stage-3 not included.

**Validated end-to-end on lab (2026-09-15)** with real data for all three sources:
AWS S3 human ITW data (Stage-1), HuggingFace `NeoteAIEmbodied/OpenNeoData` (Stage-2),
and the ready-made wetlab dataset on HF (post-train). See git history for details.

The image ships a real `git` checkout, so `git pull` inside a container picks up code
updates without a rebuild.

## Quickest path: start at Stage-2 (skip Stage-1)

We already trained Stage-1 to 14,000 steps and published it on HF, so you don't need
to run Stage-1 yourself — two scripts, two commands:

```bash
bash scripts/docker/open_container.sh   # pulls the image, opens a shell inside it
bash scripts/docker/quickstart.sh       # downloads prereqs, runs Stage-2 -> merge -> post-train
```

`open_container.sh` runs on YOUR machine (get it from this repo, or from whoever sent
you this README) — **edit the four placeholder values at its top** (`DATA_DIR`,
`CHECKPOINTS_DIR`, `ASSETS_DIR`, `HF_TOKEN`) to real local paths / your HF token before
running it. `quickstart.sh` runs once you're inside the container; both it and the two
scripts it chains (`setup_stage2_prereqs.sh`, `run_stage2_onward.sh`) accept env var
overrides (`STAGE1_STEP`, `OPENNEODATA_PLATFORM`, `EXP_NAME_STAGE2`, etc. — see each
script's top) and pass extra args straight through to the underlying `train_*.sh` (e.g.
`--num-train-steps=5` for a smoke test first). See "Run — Stage-2" / "Run — post-train"
below for what each step does individually, and the env var reference table for every
knob.

## Build

```bash
docker build . -t n0vtla_train -f scripts/docker/train.Dockerfile
```

Default CUDA base is `12.2.2-cudnn8-runtime-ubuntu22.04`. Override for other GPU
archs (e.g. Blackwell needs cu128):
```bash
docker build . -t n0vtla_train -f scripts/docker/train.Dockerfile \
  --build-arg CUDA_IMAGE_TAG=12.8.0-cudnn-runtime-ubuntu22.04
```

## Prerequisites

1. **Base checkpoint**: `hf download NeoteAI/n0-vtla-base --local-dir checkpoints/n0-vtla-base`
2. **Stage-1 data**: raw ITW corpus at `<raw_root>/<date>/<episode_uuid>/` with
   `left_hand_data.npz`, `right_hand_data.npz`, `rgb_head.{csv,mp4}`,
   `wrist_left.{csv,mp4}`, `wrist_right.{csv,mp4}`, optional `task_info.json`. All
   three camera CSVs must be real, present files — a missing view breaks the loader.
3. **Stage-2 data**: a slice of `NeoteAIEmbodied/OpenNeoData` (HF, gated), downloaded
   via `scripts/download_openneodata_flexiv_smoke.py` into canonical LeRobot-v3 layout,
   plus `norm_stats.json` under `assets/vtla_stage2_align_expert/<asset_id>/`.
4. **Post-train data**: a LeRobot-format robot dataset, plus `norm_stats.json` under
   `assets/<config_name>/<asset_id>/`. Either your own (S3), or the ready-made wetlab
   set on HF (see below).

## Run — Stage-1

```bash
docker run --rm --gpus=all \
  -v $PWD/checkpoints:/app/checkpoints \
  -v /path/to/raw_itw_root:/data/itw_raw \
  -e VTLA_ITW_RAW_ROOT=/data/itw_raw \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
  n0vtla_train bash train_stage1.sh
```
`CHECK_ONLY=1` for preflight only. On a stable external server (no wall-time
segmentation), bump checkpoint cadence: `--save-interval=5000` (CLI override only —
the code default `500` stays as-is, shared with the live VISION training config).

Or sync straight from S3 first: `n0vtla_train bash scripts/docker/run_stage1.sh
s3://bucket/prefix/itw_raw` (needs `-v ~/.aws:/root/.aws:ro`).

## Run — Stage-2

```bash
# 1. Download a slice (any OpenNeoData platform: flexiv/umi/arx5/ur/aloha/umi_single/arx5_single)
docker run --rm -e HF_TOKEN=<token> -v $PWD/data:/data n0vtla_train \
  python scripts/download_openneodata_flexiv_smoke.py --platform flexiv \
  --output /data/openneodata_smoke --num-episodes 2

# 2. Norm stats
docker run --rm -v $PWD/data:/data -v $PWD/assets:/app/assets n0vtla_train \
  python scripts/compute_canonical_norm.py --train-config-name vtla_stage2_align_expert \
  --repo-id /data/openneodata_smoke --asset-id openneodata_smoke

# 3. Train (warm-starts from base + Stage-1 checkpoint)
docker run --rm --gpus=all \
  -v $PWD/checkpoints:/app/checkpoints -v $PWD/data:/data -v $PWD/assets:/app/assets \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
  -e VTLA_STAGE1_CHECKPOINT=/app/checkpoints/vtla_stage1_predictor_pretrain/<exp_name> \
  -e VTLA_DATASET_PATH=/data/openneodata_smoke -e VTLA_ASSET_ID=openneodata_smoke \
  n0vtla_train bash train_stage2.sh

# 4. Merge before post-train (Stage-2 output is a delta, not a full checkpoint)
docker run --rm -v $PWD/checkpoints:/app/checkpoints n0vtla_train \
  python scripts/merge_stage2_checkpoint_for_posttrain.py \
  --base-checkpoint /app/checkpoints/n0-vtla-base \
  --stage1-checkpoint /app/checkpoints/vtla_stage1_predictor_pretrain/<exp_name> \
  --stage2-checkpoint /app/checkpoints/vtla_stage2_align_expert/<exp_name> \
  --output /app/checkpoints/merged_for_posttrain
```

## Run — post-train

```bash
docker run --rm --gpus=all \
  -v $PWD/checkpoints:/app/checkpoints \
  -v /path/to/lerobot_dataset:/data/robot_dataset -v $PWD/assets:/app/assets \
  -e VTLA_DATASET_PATH=/data/robot_dataset \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/merged_for_posttrain \
  n0vtla_train bash train.sh
```

**Ready-made wetlab dataset** (no conversion needed, already canonical):
```bash
# download train + holdout splits
docker run --rm -v $PWD/data:/data n0vtla_train hf download qqyang/zihiao_real_test \
  --repo-type dataset --include "n0vtla_wetlab_canonical_v2/train/**" --local-dir /data
# -> lands at /data/n0vtla_wetlab_canonical_v2/train ; repeat --include for .../holdout/**

# norm stats (train split only)
docker run --rm -v $PWD/data:/data -v $PWD/assets:/app/assets n0vtla_train \
  python scripts/compute_canonical_norm.py --train-config-name vtla_tactile_posttrain \
  --robot aloha --repo-id /data/n0vtla_wetlab_canonical_v2/train --asset-id wetlab_v2_train
# then train.sh with VTLA_DATASET_PATH=/data/n0vtla_wetlab_canonical_v2/train, VTLA_ASSET_ID=wetlab_v2_train
```

**Offline ship-gate eval** (noise-vs-signal sanity check, not an accuracy eval — pass
the SAME `VTLA_ASSET_ID` used in training, or it resolves norm stats under the wrong name):
```bash
docker run --rm --gpus=all -v $PWD/checkpoints:/app/checkpoints -v $PWD/data:/data \
  -v $PWD/assets:/app/assets -e VTLA_ASSET_ID=wetlab_v2_train n0vtla_train \
  python scripts/eval_wetlab_ship_gate.py --config vtla_tactile_posttrain \
  --checkpoint /app/checkpoints/vtla_tactile_posttrain/<exp_name>/<step> \
  --dataset-root /data/n0vtla_wetlab_canonical_v2/holdout \
  --output /app/checkpoints/ship_gate_<step>.json
```

## Env var reference

| Var | Stage-1 | Stage-2 | Post-train | Default |
|---|:-:|:-:|:-:|---|
| `VTLA_ITW_RAW_ROOT` | yes | — | — | none |
| `VTLA_ITW_NORMALIZATION` | optional | — | — | committed asset |
| `VTLA_STAGE1_CHECKPOINT` | — | yes | — | none |
| `VTLA_DATASET_PATH` | — | yes | yes | none |
| `VTLA_ASSET_ID` | — | yes | yes (else default id) | `canonical_tactile_task` |
| `VTLA_PRETRAINED_CHECKPOINT` | yes | yes | yes | none |
| `HF_TOKEN` | — | yes | optional | none |
| `CONFIG_NAME` / `EXP_NAME` | per-script defaults | | | |
| `NPROC_PER_NODE` | 8 | 8 | 8 | 8 |
| `CHECK_ONLY` | preflight-only if `1` | | | `0` |

## Troubleshooting

- **`Could not load libtorchcodec`** — FFmpeg shared libs; baked in via conda-forge.
- **DINOv2 cache miss** — don't override `HF_HOME` unless re-priming the bake-time cache.
- **`cv2` import errors** — check `opencv-python`/`opencv-python-headless` install order.
- **"A full epoch had no valid future tactile targets"** — real QC-gate failure; check
  your raw data matches the Stage-1 directory contract, not a code bug.

## Non-goals

- Downloading/preparing data is the operator's job (or use the download scripts above).
- Stage-3 packaging (separate branch, not ready).
