#!/bin/bash
#SBATCH --job-name=n0vtla-ts-compat
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/touchscale_compat_%j.log
# Mid-train on the TouchScale corpus runs from its OWN clone (N0-VTLA_touchscale), never the live
# N0-VTLA checkout other post-train jobs on this account use. This job creates/updates that clone
# (fast-forward only) and runs scripts/check_touchscale_compat.py.
set -euo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
source $YQQ/env.sh
CLONE=$YQQ/N0-VTLA_touchscale
if [[ ! -d $CLONE/.git ]]; then git clone -q https://github.com/YvonneoO/N0-VTLA.git $CLONE; fi
cd $CLONE && git pull -q --ff-only && git log -1 --format='clone at %h %s'
export PYTHONDONTWRITEBYTECODE=1
$YQQ/envs/n0vtla/bin/python scripts/check_touchscale_compat.py /scratch/project/prj-02-phai-lab/dayou/rss_27/touchscale_data \
  /scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json \
  docs/artifacts/itw_scaling_splits.json
