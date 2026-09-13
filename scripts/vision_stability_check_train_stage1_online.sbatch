#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-stability
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_stability_%j.log
#
# Longer-than-smoke-test run of the online Stage-1 trainer to check numerical
# stability over more steps -- same SMALL data footprint as the 5-step correctness
# smoke test (job 534645, 10 itw08-10 episodes -> 3 pass alignment -> 1517 samples;
# not scaling up data, per instruction). 500 steps at batch=2 is < 1 epoch over that
# sample pool, several passes over the ~3 surviving episodes -- enough to see whether
# loss/grad_norm stay finite and don't diverge, not a quality/convergence run.
#
# Submit with: sbatch scripts/vision_stability_check_train_stage1_online.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__)"

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
export VTLA_ITW_DATES=itw08-10
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/v10/per_task_scale_normalization/normalization_v10_pad30_per_task_scale.json
export VTLA_ITW_MAX_EPISODES=10
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="online_stability_${SLURM_JOB_ID}"
python scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --batch-size=2 \
  --num-workers=0 \
  --num-train-steps=500 \
  --log-interval=25 \
  --save-interval=500 \
  --overwrite

echo VISION_STAGE1_ONLINE_STABILITY_COMPLETE
