#!/bin/bash
#SBATCH --job-name=n0vtla-up-pt-scaling
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/upload_scaling_posttrain_%j.log
#
# Uploads the scaling-study post-train checkpoints (every saved step: 5000, 9999, 10000) to
# qqyang/zihiao_real_test:n0vtla_scaling_posttrain/<exp>/<step>/ . optimizer.pt (~14GB per
# checkpoint, resume state only) is excluded -- same as the earlier Task1 post-train uploads
# (model.safetensors + metadata.pt + assets/), and it keeps the total near ~200GB instead of ~540GB.
# The optimizer states stay on VISION.
#   sbatch --export=ALL,EXPS="sv2_posttrain_s1_20pct_10k sv2_posttrain_s1_40pct_10k" deploy/vision/n0vtla_upload_scaling_posttrain_ckpts_siyuan.sbatch
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
CKPT_BASE=$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain
: "${EXPS:?EXPS is required (space-separated experiment names)}"

[[ "$($HF auth whoami)" == *qqyang* ]] || { echo "ERROR: HF account is not qqyang" >&2; exit 1; }

for exp in $EXPS; do
  [[ -d "$CKPT_BASE/$exp" ]] || { echo "SKIP $exp: missing" >&2; continue; }
  echo "--- uploading $exp ---"
  $HF upload qqyang/zihiao_real_test "$CKPT_BASE/$exp" "n0vtla_scaling_posttrain/$exp" \
    --repo-type dataset --token "$HF_TOKEN" --exclude "*optimizer.pt" --exclude "*/optimizer.pt"
  echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/n0vtla_scaling_posttrain/$exp"
done
echo UPLOAD_SCALING_POSTTRAIN_DONE
