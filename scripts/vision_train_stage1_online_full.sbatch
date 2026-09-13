#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-full
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_full_%j.log
#
# Formal (not exploratory) online Stage-1 training run: full itw07-23..itw09-10
# corpus (no date/episode cap), real batch_size=64 and num_train_steps=20_000 (the
# vtla_stage1_predictor_pretrain TrainConfig's own defaults -- neither overridden
# here), the new full-corpus per-task-scale table, single node 8-GPU DDP (one full
# VISION node -- avoids the added complexity/risk of an untested multi-node rendezvous
# for this first formal launch). Gated on job 535038 (single-GPU, itw08-10-only,
# real-batch-size convergence check) showing a clean decreasing nce trend with no
# instability through step 150.
#
# Resumable: if walltime runs out before step 20000, resubmit with
# --export=ALL,EXP_NAME=<this job's exp name> and add --resume to the python call
# below (checkpoints save every save_interval=2000 steps by TrainConfig default).
#
# Submit with: sbatch scripts/vision_train_stage1_online_full.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
# No VTLA_ITW_DATES / VTLA_ITW_MAX_EPISODES -- full corpus, every date under raw_root.
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="${EXP_NAME:-online_full_${SLURM_JOB_ID}}"
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers=2 \
  --wandb-enabled

echo VISION_STAGE1_ONLINE_FULL_LAUNCHED
