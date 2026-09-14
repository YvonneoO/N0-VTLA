#!/bin/bash
#SBATCH --job-name=n0vtla-upload-stage1
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/upload_stage1_%j.log
#
# Uploads online_full_v3's CURRENT latest Stage-1 checkpoint to HuggingFace, WITHOUT
# deleting anything locally (unlike vision_prune_and_upload_stage1_ckpt.sbatch) -- for an
# ad hoc "grab the latest snapshot" upload while the run is still actively training/
# checkpointing. qqyang/zihiao_real_test is a PRE-EXISTING dataset repo (created
# 2026-09-08, not ours -- see the HF repo registry in
# Ego2Contact_project/rules/vision-workflow.md), so --repo-type MUST be dataset, never
# model (a same-named model-type repo is a separate HF namespace and would silently
# create a stray duplicate -- this bit us once on 2026-09-14).
#
# Submit with: sbatch scripts/vision_upload_latest_stage1_ckpt.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

cd /scratch/project/prj-02-phai-lab/yqq/N0-VTLA

CKPT_ROOT="checkpoints/vtla_stage1_predictor_pretrain/online_full_v3"

echo "=== checkpoints present ==="
ls -1 "$CKPT_ROOT"

# Highest-numbered checkpoint dir (digits-only names; skip any stray tmp_*).
LATEST=""
for d in "$CKPT_ROOT"/*/; do
  name=$(basename "$d")
  if [[ "$name" =~ ^[0-9]+$ ]]; then
    if [[ -z "$LATEST" || "$name" -gt "$LATEST" ]]; then
      LATEST="$name"
    fi
  fi
done
if [[ -z "$LATEST" ]]; then
  echo "No numeric checkpoint dir found under $CKPT_ROOT" >&2
  exit 1
fi
echo "Latest checkpoint: step $LATEST"

echo
echo "=== HF auth check (must be qqyang, never the cached labmate login) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting upload (env.sh not sourced first?)" >&2
  exit 1
fi

echo
echo "=== uploading checkpoint step $LATEST to qqyang/zihiao_real_test:n0-vtla_ts_pretrain/$LATEST ==="
hf upload qqyang/zihiao_real_test "$CKPT_ROOT/$LATEST" "n0-vtla_ts_pretrain/$LATEST" \
  --repo-type dataset --token "$HF_TOKEN"

echo UPLOAD_ONLY_COMPLETE
