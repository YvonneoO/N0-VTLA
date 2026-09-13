#!/bin/bash
#SBATCH --job-name=n0vtla-task-schema
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/task_schema_%j.log
#
# Standalone (no torchcodec/ffmpeg dependency): check a handful of real tujian_v5_recent
# task_info.json files to settle the "name" vs "steps" key question for per-task-scale
# normalization lookup. See scripts/inspect_task_info_schema.py.
#
# Submit with: sbatch scripts/vision_inspect_task_schema.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
python scripts/inspect_task_info_schema.py \
  "$ROOT/itw07-23/0021b7c0-b812-4ce5-9f08-31573c06977f"
echo VISION_TASK_SCHEMA_CHECK_COMPLETE
