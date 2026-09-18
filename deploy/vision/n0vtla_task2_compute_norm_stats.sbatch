#!/bin/bash
#SBATCH --job-name=n0vtla_t2_normstats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t2_normstats_%j.log
#
# Computes Task2's norm_stats.json with the ROTATION-AWARE delta fix (the same one
# applied to Task1 this session: RelRotAbsoluteActions/ChunkDeltaToCurrentState in
# transforms.py + apply_delta_rotation_aware() in compute_canonical_norm.py). Using
# --robot wetlab_right_arm (NOT --robot flexiv, which only ever touches actions[:10] --
# always-zero for wetlab data -- and does naive element-wise rot6d subtraction, both
# bugs fixed this session). Writes to
# assets/vtla_tactile_posttrain/task2_softhardsort_train/norm_stats.json
# (compute_canonical_norm.py writes there itself; no copy step needed).
#
# Submit with: sbatch --dependency=afterok:<build_jobid> deploy/vision/n0vtla_task2_compute_norm_stats.sbatch
set -euo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
N0VTLA_ENV=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$N0VTLA_ENV/bin:$PATH"

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla

python scripts/compute_canonical_norm.py \
  --repo-id "$DATA_ROOT/canonical_wetlab_task2_train" \
  --robot wetlab_right_arm \
  --train-config-name vtla_tactile_posttrain \
  --asset-id task2_softhardsort_train \
  --train-only

echo
echo "=== verifying: action[10:19] should look like a small delta, not absolute ~mm-scale ==="
python3 -c "
import json
d = json.load(open('assets/vtla_tactile_posttrain/task2_softhardsort_train/norm_stats.json'))
state = d['norm_stats']['state']['mean'][10:19]
action = d['norm_stats']['actions']['mean'][10:19]
print('state[10:19] mean:', state)
print('action[10:19] mean:', action)
xyz_small = all(abs(v) < 50 for v in action[0:3])
rot_near_identity = abs(action[3] - 1) < 0.3 and abs(action[7] - 1) < 0.3
print('action xyz looks like a small delta (not absolute state-scale):', xyz_small)
print('action rot6d looks near-identity (correct-delta signature):', rot_near_identity)
"

echo "=== TASK2_NORM_STATS_COMPLETE ==="
