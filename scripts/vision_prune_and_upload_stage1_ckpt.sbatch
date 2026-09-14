#!/bin/bash
#SBATCH --job-name=n0vtla-prune-upload-stage1
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/prune_upload_stage1_%j.log
#
# Uploads online_full_v3's latest Stage-1 checkpoint (checkpoints/vtla_stage1_predictor_pretrain/
# online_full_v3/<highest step>) to HuggingFace under qqyang/zihiao_real_test's
# n0-vtla_ts_pretrain/ folder, and ONLY AFTER that upload is confirmed successful, prunes
# the intermediate checkpoint dirs (the ones below the highest step -- they only existed so
# --resume could pick up from any of them mid-segment). Upload-before-delete deliberately:
# if the upload step fails or is interrupted, `set -e` stops the script before anything is
# deleted, so a failed/partial upload never costs us the local checkpoints too. CPU-only (no
# GPU needed for file ops + an HF upload).
#
# Submit with: sbatch scripts/vision_prune_and_upload_stage1_ckpt.sbatch
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
echo "=== ensuring qqyang/zihiao_real_test exists (repo-type model) ==="
hf repo create zihiao_real_test --repo-type model --token "$HF_TOKEN" || \
  echo "repo create skipped (likely already exists)"

echo
echo "=== uploading checkpoint step $LATEST to qqyang/zihiao_real_test:n0-vtla_ts_pretrain/$LATEST ==="
hf upload qqyang/zihiao_real_test "$CKPT_ROOT/$LATEST" "n0-vtla_ts_pretrain/$LATEST" \
  --repo-type model --token "$HF_TOKEN"
echo "Upload confirmed (hf upload returned success) -- safe to prune now."

echo
echo "=== deleting all other checkpoint dirs (upload above already succeeded) ==="
for d in "$CKPT_ROOT"/*/; do
  name=$(basename "$d")
  if [[ "$name" =~ ^[0-9]+$ && "$name" != "$LATEST" ]]; then
    du -sh "$d"
    rm -rf "$d"
    echo "removed: $d"
  fi
done

echo
echo "=== checkpoints present after pruning ==="
ls -1 "$CKPT_ROOT"

echo UPLOAD_AND_PRUNE_COMPLETE
