# N0-VTLA training image (Stage-1 + post-train)

## Scope

This image packages **Stage-1 predictor-grounding pretraining** (paper Sec 4.2,
action-free, human ITW tactile data, `train_stage1.sh` → `scripts/train_stage1_online.py`)
and **post-training** (action-conditioned, robot LeRobot-format data, `train.sh` →
`scripts/train_n0vtla.py`). **Stage-2/3 (latent↔action-expert alignment) are NOT
included** — that work is still in progress on a separate branch.

Downloading/preparing your own data (from AWS S3 or elsewhere) is entirely your own
responsibility. This image and the code inside it need nothing beyond: the image
itself, a pretrained checkpoint, and your data in the layouts documented below.
`awscli` is baked in for exactly this (see "Simplest path" below) — no credentials are
ever baked in, only the CLI tool.

The image ships a real, remote-configured `git` checkout (not a stripped-down copy),
so `git pull` inside a running container picks up a code update without a full
rebuild: `docker run --rm ... n0vtla_train bash -lc "git pull && ..."` (needs network +
repo read access at pull time, same as any other `git pull`).

## Build

```bash
docker build . -t n0vtla_train -f scripts/docker/train.Dockerfile
```

The CUDA base image defaults to `12.2.2-cudnn8-runtime-ubuntu22.04` (Ampere/Hopper).
For a different GPU architecture (e.g. Blackwell needs cu128-class images — see
`docs/PRETRAIN_IMPLEMENTATION.md`'s note that cu126 wheels had kernel issues there),
override it:

```bash
docker build . -t n0vtla_train -f scripts/docker/train.Dockerfile \
  --build-arg CUDA_IMAGE_TAG=12.8.0-cudnn-runtime-ubuntu22.04
```

Find the right tag by checking the target host's driver version (`nvidia-smi`) against
[nvidia/cuda's tag list](https://hub.docker.com/r/nvidia/cuda/tags).

`requirements.txt` pulls `lerobot` via a `pip install git+https://...` dependency —
the build environment needs outbound git/network access. If you're building this image
somewhere without that (e.g. an air-gapped target server), build it elsewhere and
`docker save`/`load` or push/pull it instead.

## Prerequisites (before `docker run`)

1. **Pretrained base checkpoint** (not baked into the image — kept as a runtime mount
   so the image isn't tied to one checkpoint version):
   ```bash
   hf download NeoteAI/n0-vtla-base --local-dir checkpoints/n0-vtla-base
   ```

2. **Stage-1 data** — your own raw ITW corpus, downloaded/unpacked from S3 into this
   exact directory contract:
   ```
   <raw_root>/<date>/<episode_uuid>/
     left_hand_data.npz       # timestamps + tactile_<pad> arrays
     right_hand_data.npz
     rgb_head.csv              # columns: frame_index, timestamp_s
     wrist_left.csv
     wrist_right.csv
     rgb_head.mp4
     wrist_left.mp4
     wrist_right.mp4
     task_info.json             # optional; falls back to "Perform the task."
   ```
   `<date>` can be any subdirectory name (it's just a partition, not parsed as a real
   date) — `train_stage1_online.py` scans every `<date>` under `<raw_root>` unless
   `VTLA_ITW_DATES` restricts it. Missing `wrist_left`/`wrist_right` views are
   tolerated (per-view masking), but at least `rgb_head` + one hand's tactile data is
   needed for a usable episode.

3. **Post-train data** — your own robot dataset in LeRobot format:
   ```
   <dataset_root>/
     meta/info.json
     meta/tasks.jsonl + meta/episodes.jsonl          # v2.1
       (or meta/tasks.parquet + meta/episodes/        # v3)
     data/chunk-000/episode_*.parquet
     videos/chunk-000/<camera_key>/episode_*.mp4
   ```
   Plus a precomputed `norm_stats.json` under
   `assets/<config_name>/<asset_id>/norm_stats.json` (default `config_name=
   vtla_tactile_posttrain`, `asset_id=canonical_tactile_task` unless overridden).

## Simplest path: have an S3 URI + AWS credentials, no local data yet

The image bakes in `awscli` (never any credentials) specifically so this is a single
command per stage — `scripts/docker/run_stage1.sh` / `run_posttrain.sh` `aws s3 sync`
your data down, then launch the matching trainer:

```bash
# Stage-1
docker run --rm --gpus=all \
  -v ~/.aws:/root/.aws:ro -v $PWD/data:/data -v $PWD/checkpoints:/app/checkpoints \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
  n0vtla_train bash scripts/docker/run_stage1.sh s3://bucket/prefix/itw_raw

# Post-train
docker run --rm --gpus=all \
  -v ~/.aws:/root/.aws:ro -v $PWD/data:/data -v $PWD/checkpoints:/app/checkpoints \
  -v $PWD/assets:/app/assets \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
  -e VTLA_ASSET_ID=my_dataset \
  n0vtla_train bash scripts/docker/run_posttrain.sh s3://bucket/prefix/robot_dataset
```

Still need the pretrained checkpoint first (`hf download NeoteAI/n0-vtla-base
--local-dir checkpoints/n0-vtla-base`, see below — separate from S3, it's public on
HF) and, for post-train, your own `norm_stats.json` under `assets/vtla_tactile_posttrain/
<asset_id>/` (not part of the raw dataset sync). Extra args after the `s3://...` pass
straight through to `train_stage1.sh`/`train.sh` (e.g. `CHECK_ONLY=1`, `--resume`).

## Run (data already local) — Stage-1

```bash
docker run --rm --gpus=all \
  -v $PWD/checkpoints:/app/checkpoints \
  -v /path/to/raw_itw_root:/data/itw_raw \
  -e VTLA_ITW_RAW_ROOT=/data/itw_raw \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
  -e NPROC_PER_NODE=8 \
  n0vtla_train bash train_stage1.sh
```

Preflight only (no training, just validates env/paths/checkpoint/DINOv2 cache):
```bash
docker run --rm --gpus=all -e CHECK_ONLY=1 ... n0vtla_train bash train_stage1.sh
```

`VTLA_ITW_NORMALIZATION` defaults to the committed
`assets/itw_normalization/normalization_v10_pad30_per_task_scale.json` — override it
only if you've fit your own table. Any task name not present in that table's
`task_scale` silently falls back to its `default_scale` (no code change needed for new
tasks — see `scripts/itw_pressure.py:normalize_pressure`).

## Run — post-train

```bash
docker run --rm --gpus=all \
  -v $PWD/checkpoints:/app/checkpoints \
  -v /path/to/lerobot_dataset:/data/robot_dataset \
  -v $PWD/assets:/app/assets \
  -e VTLA_DATASET_PATH=/data/robot_dataset \
  -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
  -e NPROC_PER_NODE=8 \
  n0vtla_train bash train.sh
```

Same `CHECK_ONLY=1` pattern applies.

## Env var reference

| Var | Stage-1 | Post-train | Required | Default |
|---|:-:|:-:|:-:|---|
| `VTLA_ITW_RAW_ROOT` | yes | — | yes | none |
| `VTLA_ITW_NORMALIZATION` | yes | — | no | committed `assets/itw_normalization/...json` |
| `VTLA_ITW_DATES` | yes | — | no | every date under `VTLA_ITW_RAW_ROOT` |
| `VTLA_ITW_MAX_EPISODES` | yes | — | no | all (smoke-test cap only) |
| `VTLA_STAGE1_FUTURE_OFFSET` | yes | — | no | 50 |
| `VTLA_STAGE1_RECON_GRID` / `_LAMBDA_REC` / `_TEMPERATURE` | yes | — | no | 8 / 0.5 / 1.0 |
| `VTLA_DATASET_PATH` | — | yes | yes | none |
| `VTLA_ASSET_ID` | — | yes | no | `canonical_tactile_task` |
| `VTLA_PRETRAINED_CHECKPOINT` | yes | yes | yes | none |
| `VTLA_DEFAULT_PROMPT` | yes | yes | no | `"Perform the task."` |
| `CONFIG_NAME` | `vtla_stage1_predictor_pretrain` | `vtla_tactile_posttrain` | — | per-script |
| `EXP_NAME` | `stage1_online` | `tactile_posttrain` | no | per-script |
| `NPROC_PER_NODE` | yes | yes | no | 8 |
| `CHECK_ONLY` | yes | yes | no | 0 |

## Troubleshooting

- **`Could not load libtorchcodec` / FFmpeg errors** — Stage-1's `.mp4` decoding
  (`torchcodec`) needs FFmpeg's shared libs; this image installs `ffmpeg<8` via
  conda-forge into the `vtla` env at build time. If you built a custom variant of this
  Dockerfile without that step, this is the first thing to check.
- **DINOv2 offline-cache miss at runtime** despite it being baked in at build time —
  almost always an `HF_HOME` mismatch: the bake step used `HF_HOME=/opt/hf_cache`
  (baked into the image via `ENV`); don't override `HF_HOME` at `docker run` time
  unless you're also re-priming the cache there.
- **`cv2` import errors** — `requirements.txt` pins both `opencv-python` and
  `opencv-python-headless`; if their install order ever produces a broken `cv2`,
  that's the first place to look (not something this Dockerfile works around).
- **`RuntimeError: A full epoch had no valid future tactile targets...`** — a
  correctness guard, not a flaky error: it means every batch in an epoch failed
  Stage-1's alignment/QC gate. Check your raw data actually matches the directory
  contract above (especially per-view CSV timestamp columns) before assuming it's a
  code bug.

## Non-goals

- Downloading or preparing data from S3 (or anywhere else) is the operator's job —
  this image assumes data is already unpacked locally in the layouts above.
- Stage-2/3 packaging (separate branch, not ready yet).
