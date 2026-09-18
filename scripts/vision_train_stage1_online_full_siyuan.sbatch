#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-siyuan
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA/stage1_online_siyuan_%j.log
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
# Chaining/override pattern identical to vision_train_stage1_online_full.sbatch:
# NPROC_PER_NODE/EXP_NAME/RESUME/NUM_WORKERS/NUM_TRAIN_STEPS via --export=ALL,...
#   sbatch --export=ALL,EXP_NAME=<name>,NUM_TRAIN_STEPS=5 scripts/vision_train_stage1_online_full_siyuan.sbatch
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw
# TODO(verify): normalization table path -- phai-lab's copy
# (haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json) is
# NOT reachable from this account (cross-account /scratch is permission-denied). Point
# this at whatever copy actually exists under this account's own yqq -- confirm the
# real path before relying on this default.
export VTLA_ITW_NORMALIZATION="${VTLA_ITW_NORMALIZATION:-/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json}"
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base

EXP_NAME="${EXP_NAME:-online_siyuan_${SLURM_JOB_ID}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers="$NUM_WORKERS" \
  --num-train-steps="$NUM_TRAIN_STEPS" \
  --wandb-enabled \
  "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo VISION_STAGE1_ONLINE_SIYUAN_LAUNCHED
