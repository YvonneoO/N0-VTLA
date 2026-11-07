#!/bin/bash
#SBATCH --job-name=n0vtla-ts-inputs
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/touchscale_inputs_%j.log
# manifest (all / old100), sample-index caches and the extended per-task normalization for mid-train on the
# TouchScale corpus -> yqq/data/touchscale_midtrain/ (scripts/build_touchscale_midtrain_inputs.py).
set -euo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
source $YQQ/env.sh
CLONE=$YQQ/N0-VTLA_touchscale
cd $CLONE && git pull -q --ff-only && git log -1 --format='clone at %h %s'
export PYTHONDONTWRITEBYTECODE=1
$YQQ/envs/n0vtla/bin/python scripts/build_touchscale_midtrain_inputs.py \
  /scratch/project/prj-02-phai-lab/dayou/rss_27/touchscale_data \
  /scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json \
  docs/artifacts/itw_scaling_splits.json $YQQ/data/touchscale_midtrain --workers 16
