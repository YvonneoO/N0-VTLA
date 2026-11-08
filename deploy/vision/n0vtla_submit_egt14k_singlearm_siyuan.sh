#!/bin/bash
# EgoTouch-init single-arm post-train (real tactile, same recipe as <task>_posttrain_ours_vision8gpu: bs64 on 8 GPUs,
# 20000 steps, cosine over 20000) as short backfill segments: per task, NSEG training segments chained afterany
# (RESUME=auto; a segment after completion exits at once), each followed (afterany) by an incremental HF upload of the
# steps that are multiples of 2000 or 19999 to qqyang/zihiao_real_test:n0vtla_singlearm_egt14k_posttrain/<exp>/.
# Runs on the VISION login node (only calls sbatch); $(...) is evaluated by THIS bash on VISION.
#   bash deploy/vision/n0vtla_submit_egt14k_singlearm_siyuan.sh "st task1 task2 task3" [NSEG=6] [FIRST_AFTER=<jobid>]
#   FIRST_AFTER: the first segment of each task waits (afterany) for that job, e.g. the last segment of another chain (GPU cap).
set -euo pipefail
TASKS="${1:?tasks, e.g. \"sv2 task1 task2 task3\"}"
NSEG="${2:-6}"
FIRST_AFTER="${3:-}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA_egotouch
PRE=$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_egotouch_14000
[[ -s "$PRE/model.safetensors" ]] || { echo "missing $PRE"; exit 1; }
for t in $TASKS; do
  exp="${t}_posttrain_egt14k_realtac_vision8gpu"
  prev="$FIRST_AFTER"
  for i in $(seq 1 "$NSEG"); do
    args=(--parsable --time=03:00:00 --time-min=01:30:00 --job-name="egt14k_${t}"
          --export="ALL,TASK=${t},SUFFIX=realtac,PRETRAINED=${PRE},EXP_NAME_OVERRIDE=${exp},STEPS=20000,LRD=20000,SAVE=1000,PRUNE_KEEP=2000,RESUME=auto")
    [[ -n "$prev" ]] && args+=(--dependency="afterany:${prev}")
    seg=$(sbatch "${args[@]}" deploy/vision/n0vtla_posttrain_predtac_siyuan.sbatch)
    up=$(sbatch --parsable --dependency="afterany:${seg}" \
      --export="ALL,EXP=${exp},PREFIX=n0vtla_singlearm_egt14k_posttrain,EVERY=2000,FINAL=20000,NOTE_FILE=deploy/vision/hf_notes/singlearm_egt14k_realtac.md" \
      deploy/vision/n0vtla_upload_posttrain_ckpts_siyuan.sbatch)
    echo "SUB|${exp} seg ${i}: train ${seg} (after ${prev:-none}), upload ${up}"
    prev="$seg"
  done
done
echo "SUB_DONE"
