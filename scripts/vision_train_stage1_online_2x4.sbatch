#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-2x4
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_2x4_%j.log
#
# Topology experiment: 8 total GPUs as 2 nodes x 4 GPUs (per user request), vs. the
# single-node 4- or 2-GPU jobs tried so far. Trades off against a whole-node 1x8
# request the way scripts/vision_train_stage1_online_full.sbatch's own history showed:
# smaller/partial-node requests schedule faster on this account's current queue than
# they did before, but any cross-node traffic (here: DDP gradient all-reduce, twice
# per step) crosses the node boundary over the cluster fabric instead of staying on
# NVLink within one node -- expect a per-step throughput hit vs. an equal-GPU-count
# single-node job, if one were schedulable.
#
# UNTESTED rendezvous path: every other job so far used `torchrun --standalone`
# (single-node only). This uses `srun` + c10d rendezvous across 2 nodes instead --
# genuinely new code for this project, submit expecting it may need a fix on the
# first attempt (e.g. rendezvous address/port, NCCL cross-node transport).
#
# 2h walltime (shorter than the single-node jobs' 4h) to bound the risk of this
# being wrong. save_interval dropped to 500 for the same reason as the 2-GPU variant:
# an unverified-throughput short window could otherwise cross very few (or zero)
# 2000-step checkpoint boundaries.
#
# Chain the same way as the other online training sbatch scripts:
#   First run:  sbatch --export=ALL,EXP_NAME=<name> scripts/vision_train_stage1_online_2x4.sbatch
#   Next link:  sbatch --dependency=afterany:<prev_jobid> \
#                 --export=ALL,EXP_NAME=<same name>,RESUME=1 \
#                 scripts/vision_train_stage1_online_2x4.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="${EXP_NAME:-online_2x4_${SLURM_JOB_ID}}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT=29500

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=4 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers=2 \
  --save-interval=500 \
  --wandb-enabled \
  "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo VISION_STAGE1_ONLINE_2X4_LAUNCHED
