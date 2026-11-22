#!/bin/bash
# Submits the scaling post-train jobs for one TASK as a strict one-at-a-time chain: each job
# starts (afterany) only when the previous one has ended, so at most one of these runs at a time.
# Runs on the VISION login node (only calls sbatch). $(...) is evaluated by THIS bash on VISION.
#   bash deploy/vision/n0vtla_submit_posttrain_chain_siyuan.sh task2 "20pct 40pct 60pct 80pct" [<first-job dependency, e.g. afterany:12345>]
set -euo pipefail
TASK="${1:?task (task1|sv2|task2)}"
LABELS="${2:?labels, e.g. \"20pct 40pct\"}"
DEP="${3:-}"
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract
prev="$DEP"
for label in $LABELS; do
  args=(--parsable --job-name="pt_${TASK}_${label%pct}" --export="ALL,TASK=${TASK},LABEL=${label}")
  [[ -n "$prev" ]] && args+=(--dependency="$prev")
  id=$(sbatch "${args[@]}" deploy/vision/n0vtla_posttrain_scaling_siyuan.sbatch 2>/dev/null)
  echo "${TASK} ${label} -> job ${id} (dependency: ${prev:-none})"
  prev="afterany:${id}"
done
