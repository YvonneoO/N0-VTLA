#!/bin/bash
#SBATCH --job-name=n0vtla_t2_dl_raw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t2_dl_raw_%j.log
#
# Downloads Task2's raw release (gated repo, qqyang's token already accepted the
# conditions) to VISION, mirroring how Task1's canonical build started -- except this
# time the raw->canonical conversion runs on VISION too (see n0vtla_task2_build_canonical.sbatch),
# not on lab. 2998 files across 60 episodes; CPU-only, generous 3h budget since actual
# size wasn't checked ahead of time (gated dataset, listing needs the qqyang token).
#
# Submit with: sbatch deploy/vision/n0vtla_task2_download_raw.sbatch
set -euo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
RAW_DEST="$DATA_ROOT/wetlab_raw/task2_soft_hard_sorting_20260918"
mkdir -p "$RAW_DEST"

echo "=== HF auth check (must be qqyang, gated repo needs accepted conditions) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting (env.sh not sourced first?)" >&2
  exit 1
fi

echo
echo "=== downloading task2_soft_hard_sorting_20260918 raw release ==="
hf download SingleBicycle/tacwam-wetlab-tasks --repo-type dataset \
  --include "task2_soft_hard_sorting_20260918/**" \
  --local-dir "$DATA_ROOT/wetlab_raw"

echo
echo "=== verifying ==="
ls "$RAW_DEST" | head -5
echo "episode dirs (36-char uuid names):"
ls "$RAW_DEST" | grep -cE "^[0-9a-f-]{36}$"
echo "=== TASK2_RAW_DOWNLOAD_COMPLETE ==="
