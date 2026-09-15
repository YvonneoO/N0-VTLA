#!/bin/bash
#SBATCH --job-name=n0vtla-merge-stage1-base
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/merge_stage1_base_%j.log
#
# Runs scripts/merge_stage1_into_base_checkpoint.py -- CPU-only (pure tensor I/O, no GPU/model
# forward needed). See that script's docstring for why the merge is necessary before
# post-training from a Stage-1 trainable-only checkpoint.
#
# Override EXP_NAME/STEP/OUT_NAME via --export=ALL,... ; defaults match the current Stage-1
# online_full_v3 run's latest uploaded checkpoint (step 14000).
#   sbatch scripts/vision_merge_stage1_into_base.sbatch
#   sbatch --export=ALL,EXP_NAME=online_full_v3,STEP=20000 scripts/vision_merge_stage1_into_base.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

cd /scratch/project/prj-02-phai-lab/yqq/N0-VTLA

EXP_NAME="${EXP_NAME:-online_full_v3}"
STEP="${STEP:-14000}"
OUT_NAME="${OUT_NAME:-n0-vtla-base_plus_${EXP_NAME}_${STEP}}"

python scripts/merge_stage1_into_base_checkpoint.py \
  --base-checkpoint checkpoints/n0-vtla-base \
  --stage1-checkpoint "checkpoints/vtla_stage1_predictor_pretrain/${EXP_NAME}/${STEP}" \
  --output "checkpoints/${OUT_NAME}"

echo MERGE_COMPLETE
echo "For post-training, set: VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/${OUT_NAME}"
