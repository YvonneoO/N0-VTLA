#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-smoke
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_smoke_%j.log
#
# Real end-to-end smoke test of scripts/train_stage1_online.py: warm-starts from the
# real n0-vtla-base checkpoint, trains a few real steps on real itw08-10 episodes
# (inside the per-task-scale manifest's own fit range -- confirmed 100% task_scale
# coverage there, job 534585) through the online loader, single GPU, plain python (no
# torchrun, matching how the lab's offline Stage-1 job was launched). This is
# deliberately tiny (few episodes, few steps, small batch) -- a correctness check that
# the training entrypoint runs cleanly end to end, not a throughput or convergence run.
#
# Submit with: sbatch scripts/vision_smoke_test_train_stage1_online.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
source /scratch/project/prj-02-phai-lab/yqq/env.sh

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
export VTLA_ITW_DATES=itw08-10
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/v10/per_task_scale_normalization/normalization_v10_pad30_per_task_scale.json
export VTLA_ITW_MAX_EPISODES=10
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

# wandb_enabled already defaults to False on this TrainConfig -- not overridden here.
EXP_NAME="online_smoke_${SLURM_JOB_ID}"
python scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --batch-size=2 \
  --num-workers=0 \
  --num-train-steps=5 \
  --log-interval=1 \
  --save-interval=5 \
  --overwrite

echo VISION_STAGE1_ONLINE_SMOKE_COMPLETE
