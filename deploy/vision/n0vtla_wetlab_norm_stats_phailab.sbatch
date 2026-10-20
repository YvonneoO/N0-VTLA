#!/bin/bash
#SBATCH --job-name=n0vtla_normstats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=prj-02-phai-lab
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/%x_%j.log
#
# State / action normalisation statistics (assets/vtla_tactile_posttrain/<ASSET>/norm_stats.json) of a canonical post-train dataset, train split only.
# They depend on state and action only, so every tactile variant of the same episodes gives the same numbers.
#   sbatch --export=ALL,DATASET=<.../canonical_wetlab_task3_zerotac0_train>,ASSET=task3_whiteboard_train deploy/vision/n0vtla_wetlab_norm_stats_phailab.sbatch
set -euo pipefail
YQQ="${YQQ:-/scratch/project/prj-02-phai-lab/yqq}"
: "${DATASET:?DATASET=canonical train dataset dir}"; : "${ASSET:?ASSET=asset id}"
module load Miniforge3/25.3.0-3
source $YQQ/env.sh
N0VTLA_ENV=$YQQ/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV" CONDA_DEFAULT_ENV=n0vtla PATH="$N0VTLA_ENV/bin:$PATH"
cd $YQQ/N0-VTLA
python scripts/compute_canonical_norm.py --repo-id "$DATASET" --robot wetlab_right_arm --train-config-name vtla_tactile_posttrain --asset-id "$ASSET" --train-only
python3 -c "
import json
d = json.load(open('assets/vtla_tactile_posttrain/$ASSET/norm_stats.json'))
print('state[10:19] mean:', [round(v, 3) for v in d['norm_stats']['state']['mean'][10:19]])
print('action[10:19] mean:', [round(v, 3) for v in d['norm_stats']['actions']['mean'][10:19]])
"
echo NORM_STATS_DONE
