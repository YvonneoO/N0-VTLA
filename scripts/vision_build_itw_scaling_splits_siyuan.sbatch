#!/bin/bash
#SBATCH --job-name=n0vtla-build-scaling-splits
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/build_scaling_splits_%j.log
#
# One-off CPU-only job: builds the fixed nested 20/40/60/80% ITW episode-fraction
# manifest (scripts/build_itw_scaling_splits.py) for the Stage-1 human-data
# pretraining scaling study.
#
# Runs against a FRESH, separate clone (yqq/N0-VTLA_scaling_extract), not the live
# yqq/N0-VTLA checkout: 3 training jobs are currently running against that live
# checkout, and a `git pull` there mid-run risks a respawned dataloader worker
# picking up new code while the main process still runs the old shape (see project
# memory vision-no-git-pull-during-running-job). This job's clone is fully
# independent, so it's safe to fetch main regardless of what's running elsewhere.
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh

REPO_DIR=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch origin main
  git -C "$REPO_DIR" reset --hard origin/main
else
  git clone --branch main https://github.com/YvonneoO/N0-VTLA.git "$REPO_DIR"
fi

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

cd "$REPO_DIR"
python scripts/build_itw_scaling_splits.py \
  --raw-root /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw \
  --output /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json \
  --overwrite

echo BUILD_SCALING_SPLITS_DONE
