#!/usr/bin/env bash
set -euo pipefail

# Stage-2 latent-to-expert alignment launcher (paper Sec 4.2 Stage 2) -- sibling to
# train_stage1.sh/train.sh. Mirrors their structure, but its preflight is NEW rather
# than copied from either sibling: Stage-2 uniquely needs a TWO-LINK checkpoint chain
# (base policy checkpoint AND Stage-1's own trainable-only checkpoint, both loaded
# unconditionally every launch -- see scripts/train_stage2_align_expert.py's module
# docstring) plus a canonical-schema LeRobot dataset (any OpenNeoData platform, not
# just flexiv -- see the vtla_stage2_align_expert config's 2026-09-15 comment in
# n0vtla/training/config.py) and precomputed norm stats for it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONFIG_NAME="${CONFIG_NAME:-vtla_stage2_align_expert}"
EXP_NAME="${EXP_NAME:-stage2_align}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
  [[ -z "$NPROC_PER_NODE" || "$NPROC_PER_NODE" == "0" ]] && NPROC_PER_NODE=8
fi
CHECK_ONLY="${CHECK_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash train_stage2.sh [additional training arguments]

Environment variables:
  CONFIG_NAME=vtla_stage2_align_expert
  EXP_NAME=stage2_align
  NPROC_PER_NODE=<auto-detected via nvidia-smi, falls back to 8>
  CHECK_ONLY=0
  VTLA_PRETRAINED_CHECKPOINT=/path/to/checkpoint   (required -- base policy, e.g. n0-vtla-base)
  VTLA_STAGE1_CHECKPOINT=/path/to/stage1_ckpt_dir  (required -- Stage-1's own checkpoint dir;
                                                     the latest numeric step subdir is used)
  VTLA_DATASET_PATH=/path/to/openneodata_slice     (required -- canonical LeRobot-v3 dir, any
                                                     OpenNeoData platform, see
                                                     scripts/download_openneodata_flexiv_smoke.py)
  VTLA_ASSET_ID=<id>                                (required -- norm-stats asset id, see
                                                     scripts/compute_canonical_norm.py)

Examples:
  CHECK_ONLY=1 bash train_stage2.sh
  VTLA_PRETRAINED_CHECKPOINT=checkpoints/n0-vtla-base \
  VTLA_STAGE1_CHECKPOINT=checkpoints/vtla_stage1_predictor_pretrain/stage1_online \
  VTLA_DATASET_PATH=data/openneodata_smoke VTLA_ASSET_ID=openneodata_smoke \
  bash train_stage2.sh --batch-size=2 --num-train-steps=5 --save-interval=5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EXTRA_TRAIN_ARGS=("$@")

cd "$REPO_ROOT"

PYTHON_BIN="$(command -v python)"
TORCHRUN_BIN="$(command -v torchrun)"
if [[ -z "$PYTHON_BIN" || -z "$TORCHRUN_BIN" ]]; then
  echo "python or torchrun is unavailable in the active environment" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export N0VTLA_DATA_HOME="${N0VTLA_DATA_HOME:-$REPO_ROOT/models}"
export HF_HOME="${HF_HOME:-$N0VTLA_DATA_HOME/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VTLA_ATTN_IMPL="${VTLA_ATTN_IMPL:-eager}"
export VTLA_PREFIX_CACHE="${VTLA_PREFIX_CACHE:-1}"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}.log"

echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TORCHRUN_BIN=$TORCHRUN_BIN"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "VTLA_PRETRAINED_CHECKPOINT=${VTLA_PRETRAINED_CHECKPOINT:-<unset>}"
echo "VTLA_STAGE1_CHECKPOINT=${VTLA_STAGE1_CHECKPOINT:-<unset>}"
echo "VTLA_DATASET_PATH=${VTLA_DATASET_PATH:-<unset>}"
echo "VTLA_ASSET_ID=${VTLA_ASSET_ID:-<unset>}"
echo "LOG_FILE=$REPO_ROOT/$LOG_FILE"

"$PYTHON_BIN" - "$CONFIG_NAME" "$NPROC_PER_NODE" <<'PY'
import os
import pathlib
import sys

import torch
from transformers import AutoConfig

from n0vtla.training import config as config_module

config_name = sys.argv[1]
required_gpus = int(sys.argv[2])
cfg = config_module.get_config(config_name)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this container")
visible_gpus = torch.cuda.device_count()
if visible_gpus < required_gpus:
    raise SystemExit(f"only {visible_gpus} GPUs are visible; NPROC_PER_NODE={required_gpus}")

# Link 1: base policy checkpoint.
base_checkpoint = pathlib.Path(cfg.pytorch_weight_path or "") / "model.safetensors"
if not base_checkpoint.exists():
    raise SystemExit(f"missing base checkpoint: {base_checkpoint}")

# Link 2: Stage-1's own trainable-only checkpoint -- latest numeric step subdir, same
# resolution train_stage2_align_expert.py itself uses.
stage1_dir = os.environ.get("VTLA_STAGE1_CHECKPOINT")
if not stage1_dir or not pathlib.Path(stage1_dir).is_dir():
    raise SystemExit(f"missing/invalid VTLA_STAGE1_CHECKPOINT: {stage1_dir!r}")
stage1_steps = [int(d.name) for d in pathlib.Path(stage1_dir).iterdir()
                if d.is_dir() and d.name.isdigit()]
if not stage1_steps:
    raise SystemExit(f"VTLA_STAGE1_CHECKPOINT has no numeric step subdirs: {stage1_dir}")
stage1_latest = pathlib.Path(stage1_dir) / str(max(stage1_steps)) / "model.safetensors"
if not stage1_latest.exists():
    raise SystemExit(f"missing Stage-1 checkpoint file: {stage1_latest}")

# Canonical LeRobot-v3 dataset(s) (any OpenNeoData platform(s)) -- shallow structure
# check. VTLA_DATASET_PATH may be one path or a comma-separated list of several (e.g.
# one per platform) -- LeRobotCanonicalTaskTactileDataConfig.repo_ids concatenates all
# of them for training when more than one is given.
dataset_path = os.environ.get("VTLA_DATASET_PATH")
if not dataset_path:
    raise SystemExit("missing VTLA_DATASET_PATH")
dataset_dirs = [pathlib.Path(p) for p in dataset_path.split(",")]
for dataset_dir in dataset_dirs:
    for marker in ("meta", "data", "videos"):
        if not (dataset_dir / marker).is_dir():
            raise SystemExit(f"VTLA_DATASET_PATH missing expected LeRobot-v3 dir: {dataset_dir / marker}")

# Precomputed norm stats for VTLA_ASSET_ID (scripts/compute_canonical_norm.py's output).
asset_id = os.environ.get("VTLA_ASSET_ID")
if not asset_id:
    raise SystemExit("missing VTLA_ASSET_ID")
norm_stats = pathlib.Path(cfg.assets_base_dir) / cfg.name / asset_id / "norm_stats.json"
if not norm_stats.exists():
    raise SystemExit(f"missing norm stats: {norm_stats} "
                      f"(run scripts/compute_canonical_norm.py --train-config-name {cfg.name} first)")

AutoConfig.from_pretrained("facebook/dinov2-base", local_files_only=True)

print(f"environment OK: torch={torch.__version__}, GPUs={visible_gpus}")
print(f"config OK: {cfg.name}")
print(f"base checkpoint OK: {base_checkpoint}")
print(f"Stage-1 checkpoint OK: {stage1_latest} (step {max(stage1_steps)})")
print(f"dataset OK: {dataset_dirs}")
print(f"norm stats OK: {norm_stats}")
print("DINOv2 cache OK")
PY

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "CHECK_ONLY=1; all checks passed."
  exit 0
fi

echo "$(date '+%F %T') starting training"
"$TORCHRUN_BIN" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_stage2_align_expert.py \
  "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
echo "$(date '+%F %T') training completed"
