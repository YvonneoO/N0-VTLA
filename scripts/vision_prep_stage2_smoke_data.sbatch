#!/bin/bash
#SBATCH --job-name=n0vtla-stage2-data-prep
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=stage2_data_prep_%j.log
#
# Downloads a small (--num-episodes) local slice of NeoteAIEmbodied/OpenNeoData's flexiv
# platform and computes its norm_stats.json, for smoke-testing scripts/train_stage2_align_
# expert.py. CPU-only. Run from inside the isolated Stage-2 git worktree (NOT the main
# checkout that the online_full_v3 Stage-1 chain depends on) -- this script's #SBATCH --output
# is a relative path deliberately, so the log lands in whichever worktree submitted it.
#
# Submit with (from the worktree root): sbatch scripts/vision_prep_stage2_smoke_data.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

DATA_OUT="${DATA_OUT:-/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/data/openneodata_flexiv_smoke}"
NUM_EPISODES="${NUM_EPISODES:-2}"

echo "=== HF auth check (must be qqyang) ==="
hf auth whoami

echo
echo "=== downloading OpenNeoData flexiv smoke slice ==="
python scripts/download_openneodata_flexiv_smoke.py \
  --output "$DATA_OUT" --num-episodes "$NUM_EPISODES" --overwrite

echo
echo "=== computing norm_stats.json ==="
python scripts/compute_canonical_norm.py \
  --repo-id "$DATA_OUT" \
  --robot flexiv \
  --train-config-name vtla_stage2_align_expert \
  --asset-id openneodata_flexiv_smoke

echo STAGE2_DATA_PREP_COMPLETE
