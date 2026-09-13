#!/bin/bash
#SBATCH --job-name=n0vtla-task-name-compare
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/task_name_compare_%j.log
#
# Checks whether itw's task-name vocabulary (task_info.json["name"]) actually differs
# between July and August dates, or whether the earlier 0% per-task-scale coverage on
# itw07-23 (job 534584) was just an unlucky small sample. Per-task-scale is a
# date-independent concept in principle -- this compares real sampled task names
# across a broad set of July vs August dates directly, not just against one
# audit table's coverage.
#
# Submit with: sbatch scripts/vision_compare_task_names.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
python scripts/compare_task_names_across_dates.py "$RAW_ROOT" \
  --dates-a itw07-23,itw07-25,itw07-27,itw07-28,itw07-29,itw07-30,itw07-31 \
  --dates-b itw08-03,itw08-06,itw08-07,itw08-10,itw08-11,itw08-12 \
  --sample 500
