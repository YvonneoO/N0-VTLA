#!/bin/bash
#SBATCH --job-name=n0vtla_fitnorm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=prj-02-phai-lab
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/%x_%j.log
#
# Fit the per-pad tactile normalisation (train split only, seed 42: deterministic) of Whiteboard erasing (task3) from the raw release, the same call
# n0vtla_task3_build_canonical.sbatch makes: -> $YQQ/data/n0vtla/wetlab_tactile_norm_task3_v1.json
#   sbatch deploy/vision/n0vtla_wetlab_fit_norm_phailab.sbatch
set -euo pipefail
YQQ="${YQQ:-/scratch/project/prj-02-phai-lab/yqq}"
module load Miniforge3/25.3.0-3
source $YQQ/env.sh
N0VTLA_ENV=$YQQ/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV" CONDA_DEFAULT_ENV=n0vtla PATH="$N0VTLA_ENV/bin:$PATH"
cd $YQQ/N0-VTLA
DATA=$YQQ/data/n0vtla
OUT=$DATA/wetlab_tactile_norm_task3_v1.json
if [ -f "$OUT" ]; then echo "exists: $OUT"; else
  python scripts/fit_wetlab_tactile_norm.py $DATA/wetlab_raw/task3_whiteboard_erasing_20260918 scripts/wetlab_split_task3_v1.json $OUT --task-name whiteboard_erase_v1 --skip-success-label-check
fi
python3 -c "
import json
d = json.load(open('$OUT'))
s = d['normal_scale'][15:30]
print('scale (right-hand pads):', [round(x, 5) for x in s])
print('pads at the scale floor:', [i for i, x in enumerate(s) if x <= 1.01 * min(s)])
"
echo FIT_NORM_DONE
