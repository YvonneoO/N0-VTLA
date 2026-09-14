#!/bin/bash
#SBATCH --job-name=n0vtla-prune-upload-stage1
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/prune_upload_stage1_%j.log
#
# Prunes online_full_v3's Stage-1 checkpoints (checkpoints/vtla_stage1_predictor_pretrain/
# online_full_v3/) down to the single highest-numbered one -- the intermediate saves from
# segment 1 (job 537966: 500/1000/1500/2000/2500/3000) only existed so --resume could pick
# up from any of them; once the segment ended we only need the latest -- then uploads that
# surviving checkpoint to HuggingFace under qqyang/zihiao_real_test's n0-vtla_ts_pretrain/
# folder. CPU-only (no GPU needed for file ops + an HF upload).
#
# Submit with: sbatch scripts/vision_prune_and_upload_stage1_ckpt.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

cd /scratch/project/prj-02-phai-lab/yqq/N0-VTLA

CKPT_ROOT="checkpoints/vtla_stage1_predictor_pretrain/online_full_v3"

echo "=== checkpoints present before pruning ==="
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
echo "=== deleting all other checkpoint dirs ==="
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

echo PRUNE_AND_UPLOAD_COMPLETE
