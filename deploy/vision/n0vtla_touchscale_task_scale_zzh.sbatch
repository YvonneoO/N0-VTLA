#!/bin/bash
#SBATCH --job-name=n0vtla-ts-taskscale
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/touchscale_taskscale_%j.log
# Per-task tactile scales for TouchScale tasks missing from the original table, with tacWAM's own algorithm
# (scripts/extend_task_scale_touchscale.py) -> yqq/data/touchscale_midtrain/normalization_..._tacwam.json.
# No git pull here: the clone may be in use by a running mid-train job; pull it beforehand.
set -euo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
source $YQQ/env.sh
CLONE=$YQQ/N0-VTLA_touchscale
cd $CLONE && git log -1 --format='clone at %h %s'
export PYTHONDONTWRITEBYTECODE=1
$YQQ/envs/n0vtla/bin/python scripts/extend_task_scale_touchscale.py \
  /scratch/project/prj-02-phai-lab/dayou/rss_27/touchscale_data \
  /scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json \
  $YQQ/data/touchscale_midtrain/normalization_pad30_per_task_scale_touchscale_tacwam.json --workers 16
