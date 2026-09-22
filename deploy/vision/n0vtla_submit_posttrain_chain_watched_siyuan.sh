#!/bin/bash
# Same one-at-a-time training chain as n0vtla_submit_posttrain_chain_siyuan.sh, but also submits
# a per-point upload watcher job (afterok on that training job only, so a failed point isn't
# uploaded) right after each training job -- ckpts reach HF as soon as each point finishes,
# instead of waiting for the whole TASK to complete before one big upload.
# Runs on the VISION login node (only calls sbatch). $(...) is evaluated by THIS bash on VISION.
#   bash deploy/vision/n0vtla_submit_posttrain_chain_watched_siyuan.sh task3 "20pct 40pct 60pct 80pct" [<first-job dependency>]
set -euo pipefail
TASK="${1:?task (task1|sv2|task2|task3)}"
LABELS="${2:?labels, e.g. \"20pct 40pct\"}"
DEP="${3:-}"
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract
prev="$DEP"
for label in $LABELS; do
  exp="${TASK}_posttrain_s1_${label}_10k"
  args=(--parsable --job-name="pt_${TASK}_${label%pct}" --export="ALL,TASK=${TASK},LABEL=${label}")
  [[ -n "$prev" ]] && args+=(--dependency="$prev")
  train_id=$(sbatch "${args[@]}" deploy/vision/n0vtla_posttrain_scaling_siyuan.sbatch 2>/dev/null)
  up_id=$(sbatch --parsable --job-name="up_${TASK}_${label%pct}" --dependency="afterok:${train_id}" \
    --export="ALL,EXPS=${exp}" deploy/vision/n0vtla_upload_scaling_posttrain_ckpts_siyuan.sbatch 2>/dev/null)
  echo "${TASK} ${label} -> train ${train_id} (dep: ${prev:-none}), upload watcher ${up_id} (dep: afterok:${train_id})"
  prev="afterany:${train_id}"
done
