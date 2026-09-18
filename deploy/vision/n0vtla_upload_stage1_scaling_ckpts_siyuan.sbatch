#!/bin/bash
#SBATCH --job-name=n0vtla-upload-stage1-scaling
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/upload_stage1_scaling_%j.log
#
# Uploads ONLY the FINAL (num_train_steps) checkpoint of each Stage-1 data-scaling
# run -- not every intermediate save_interval step, unlike
# n0vtla_upload_task1_vision_all_ckpts.sbatch's pattern for post-train. Destination is
# a NEW dataset folder (n0vtla_pretrain_scaling_study/), kept separate from the
# existing n0-vtla_ts_pretrain/ (the old 100%-data phai-lab run) and
# n0-vtla-base_plus_stage1_human_14000/ (that run's merged-with-base final ckpt) so
# this scaling study's artifacts never get mixed with the earlier ones.
#
# EXPS defaults to the 3 already-complete fractions; override via --export=ALL,EXPS=...
# once 80pct finishes too (space-separated "expname:label" pairs).
#   sbatch --export=ALL,EXPS="stage1_online_80pct:80pct" deploy/vision/n0vtla_upload_stage1_scaling_ckpts_siyuan.sbatch
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract

echo "=== HF auth check (must be qqyang) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting (env.sh not sourced first?)" >&2
  exit 1
fi

HF_DEST_ROOT="n0vtla_pretrain_scaling_study"
CKPT_BASE="checkpoints/vtla_stage1_predictor_pretrain"
EXPS="${EXPS:-stage1_online_20pct:20pct stage1_online_40pct:40pct stage1_online_60pct:60pct}"

for pair in $EXPS; do
  exp_name="${pair%%:*}"
  label="${pair##*:}"
  exp_dir="${CKPT_BASE}/${exp_name}"
  if [[ ! -d "$exp_dir" ]]; then
    echo "SKIP $label: $exp_dir does not exist" >&2
    continue
  fi
  latest=$(ls -1 "$exp_dir" | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [[ -z "$latest" ]]; then
    echo "SKIP $label: no numeric checkpoint step found in $exp_dir" >&2
    continue
  fi
  echo "--- uploading $label step $latest -> qqyang/zihiao_real_test:${HF_DEST_ROOT}/${label}/${latest} ---"
  hf upload qqyang/zihiao_real_test "${exp_dir}/${latest}" "${HF_DEST_ROOT}/${label}/${latest}" \
    --repo-type dataset --token "$HF_TOKEN"
  echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/${HF_DEST_ROOT}/${label}/${latest}"
done

echo VISION_UPLOAD_STAGE1_SCALING_DONE
