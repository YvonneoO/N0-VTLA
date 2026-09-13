#!/bin/bash
#SBATCH --job-name=n0vtla-no-tactile-stage1
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/no_tactile_stage1_%j.log
#
# CPU-only (no GPU/DDP, see materialize_no_tactile_stage1_checkpoint.py's docstring for
# why): materializes the no-tactile control arm's "Stage-1 output" as a zero-perturbation
# copy of n0-vtla-base's tactile-pathway tensors, in the same trainable_only_v1 checkpoint
# format the real human-data Stage-1 run produces -- so Stage-2/3 tooling can point at
# both arms' checkpoint directories uniformly. Never trains anything.
#
# Submit with: sbatch scripts/vision_materialize_no_tactile_stage1.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

cd /scratch/project/prj-02-phai-lab/yqq/N0-VTLA

export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="${EXP_NAME:-no_tactile_control}"

python scripts/materialize_no_tactile_stage1_checkpoint.py --exp-name="$EXP_NAME"

echo NO_TACTILE_STAGE1_MATERIALIZED
