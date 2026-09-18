#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-siyuan
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/stage1_online_siyuan_%j.log
#
# Same Stage-1 online-loader training as scripts/vision_train_stage1_online_full.sbatch
# (see that file's own header for the design rationale), adapted to run under the
# siyuanxu_tamu.edu / prj-02-uq-llms-for-reasoning account instead of prj-02-phai-lab --
# accounts cannot cross-read each other's /scratch, so every path below points at this
# account's OWN copies (data staged separately under this account's yqq/data/raw,
# checkpoint at yqq/data/n0vtla/n0-vtla-base), not phai-lab's.
#
# Default --time=00:20:00 is a SMOKE-TEST walltime (short on purpose) -- override with
# --time=... on the sbatch command line for a real run once the smoke test passes.
#
# Runs from yqq/N0-VTLA_scaling_extract (a clone independent of the live yqq/N0-VTLA
# checkout other jobs on this account use), not yqq/N0-VTLA, so a future git pull for
# this scaling study never risks a mid-run code-shape mismatch on unrelated jobs --
# see project memory vision-no-git-pull-during-running-job.
#
# Defaults to the 20% data-scaling split (EPISODE_LIST_KEY=20pct); override
# EPISODE_LIST_KEY=40pct/60pct/80pct for the other scaling-study points, or unset
# VTLA_ITW_EPISODE_LIST_JSON entirely to fall back to the full raw corpus.
#
# Chaining/override pattern identical to vision_train_stage1_online_full.sbatch:
# NPROC_PER_NODE/EXP_NAME/RESUME/NUM_WORKERS/NUM_TRAIN_STEPS/EPISODE_LIST_KEY via --export=ALL,...
#   sbatch --time=12:00:00 --export=ALL,EXP_NAME=<name>,NUM_TRAIN_STEPS=14000 scripts/vision_train_stage1_online_full_siyuan.sbatch
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
# Confirmed root cause 2026-09-17: job 561095 (60pct run) saved checkpoints every 500
# steps despite config.py's save_interval being 1000 on disk -- get_config()/cli()
# both returned 1000 when checked directly, but the checkpoint's OWN embedded
# metadata (dataclasses.asdict(config), the ground truth for what that exact process
# used) showed save_interval=500. A stale n0vtla/training/__pycache__/config.*.pyc
# (this clone is shared across many compute-node job launches on a parallel
# filesystem, where mtime-based bytecode-cache invalidation can go stale) is the
# leading suspect. Disabling bytecode caching entirely removes this whole bug class.
export PYTHONDONTWRITEBYTECODE=1

NORM_PATH=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json
if [[ ! -f "$NORM_PATH" ]]; then
  mkdir -p "$(dirname "$NORM_PATH")"
  hf download qqyang/zihiao_real_test \
    n0vtla_itw_normalization/normalization_pad30_per_task_scale.json \
    --repo-type dataset --token "$HF_TOKEN" \
    --local-dir /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_norm_download
  mv /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_norm_download/n0vtla_itw_normalization/normalization_pad30_per_task_scale.json "$NORM_PATH"
  rm -rf /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_norm_download
fi

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw
export VTLA_ITW_NORMALIZATION="$NORM_PATH"
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base
export VTLA_ITW_EPISODE_LIST_JSON="${VTLA_ITW_EPISODE_LIST_JSON:-/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json}"
export VTLA_ITW_EPISODE_LIST_KEY="${EPISODE_LIST_KEY:-20pct}"

EXP_NAME="${EXP_NAME:-online_siyuan_${SLURM_JOB_ID}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers="$NUM_WORKERS" \
  --num-train-steps="$NUM_TRAIN_STEPS" \
  --wandb-enabled \
  "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo VISION_STAGE1_ONLINE_SIYUAN_LAUNCHED
