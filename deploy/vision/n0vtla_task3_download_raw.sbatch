#!/bin/bash
#SBATCH --job-name=n0vtla_t3_dl_raw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t3_dl_raw_%j.log
#
# Downloads Task3's raw release (gated repo) to VISION; raw->canonical build runs on VISION
# (n0vtla_task3_build_canonical.sbatch). CPU-only, 3h budget.
#
# Submit with: sbatch deploy/vision/n0vtla_task3_download_raw.sbatch
set -euo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
# conda activation is unreliable in non-interactive sbatch shells on some nodes; PATH-bypass.
N0VTLA_ENV=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$N0VTLA_ENV/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
RAW_DEST="$DATA_ROOT/wetlab_raw/task3_whiteboard_erasing_20260918"
mkdir -p "$RAW_DEST"

echo "=== HF auth check (must be qqyang, gated repo needs accepted conditions) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting (env.sh not sourced first?)" >&2
  exit 1
fi

echo
echo "=== downloading task3_whiteboard_erasing_20260918 raw release ==="
hf download SingleBicycle/tacwam-wetlab-tasks --repo-type dataset \
  --include "task3_whiteboard_erasing_20260918/**" \
  --local-dir "$DATA_ROOT/wetlab_raw"

echo
echo "=== verifying ==="
ls "$RAW_DEST" | head -5
echo "episode dirs (36-char uuid names):"
ls "$RAW_DEST" | grep -cE "^[0-9a-f-]{36}$"
echo "=== TASK3_RAW_DOWNLOAD_COMPLETE ==="
