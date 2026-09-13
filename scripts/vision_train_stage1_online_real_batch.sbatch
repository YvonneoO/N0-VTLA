#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-realbs
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_realbs_%j.log
#
# Real-batch-size (64, the vtla_stage1_predictor_pretrain default) convergence check on
# the ONLINE loader, single GPU -- deliberately mirrors the lab's already-completed
# OFFLINE (materialized-dataset) 2000-step temp=1.0 run exactly on batch_size/world_size
# (lab log: "stage1: batch_size=64 (per-GPU=64) world_size=1") so the two nce/recon/
# grad_norm curves are directly comparable: does the online loader's real data (whole
# itw08-10 date, ~818 episodes, not the smoke tests' 10-episode cap) converge the same
# way as the offline materialized path did?
#
# 1000 steps (half of lab's 2000) to keep this exploratory run's wall-clock bounded --
# real-time video decode (torchcodec) is slower per-sample than the offline path's
# pre-materialized reads (~0.3s/sample observed in smoke tests), so this will run
# noticeably slower per-step than lab's offline run despite the same batch size.
# num_workers=2 (the max this job's CPU budget supports at the mandatory 2 CPU/GPU
# ratio) to parallelize that decode instead of serializing it in the main process.
#
# Submit with: sbatch scripts/vision_train_stage1_online_real_batch.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
export VTLA_ITW_DATES=itw08-10
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/v10/per_task_scale_normalization/normalization_v10_pad30_per_task_scale.json
# No VTLA_ITW_MAX_EPISODES -- use every itw08-10 episode that passes alignment (~818
# recordings), not the smoke tests' 10-episode cap.
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="online_realbs_${SLURM_JOB_ID}"
python scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers=2 \
  --num-train-steps=1000 \
  --log-interval=25 \
  --save-interval=250 \
  --wandb-enabled \
  --overwrite

echo VISION_STAGE1_ONLINE_REALBS_COMPLETE
