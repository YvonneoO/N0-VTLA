#!/bin/bash
# Single-arm post-train from a merged mid-train init (real tactile, same recipe as <task>_posttrain_ours_vision8gpu: bs64 on
# 8 GPUs, 20000 steps, cosine over 20000) as short backfill segments: per task, NSEG training segments chained afterany
# (RESUME=auto; a segment after completion exits at once), each followed (afterany) by an incremental HF upload to
# qqyang/zihiao_real_test:<HFPREFIX>/<exp>/ of the steps that are multiples of UP_EVERY or UP_FINAL-1.
# Runs on the VISION login node (only calls sbatch); $(...) is evaluated by THIS bash on VISION.
#   bash deploy/vision/n0vtla_submit_egt14k_singlearm_siyuan.sh "<groups>" [NSEG=6] [FIRST_AFTER=<jobid>|<type:jobid>]
#   <groups>: space-separated; tasks joined by "+" run one after another in ONE chain (e.g. "task1+task3 task2 st" = 3 chains).
#   FIRST_AFTER: the first segment of each chain waits for that job (plain id = afterany; "afterok:<id>" e.g. a smoke).
# Env (defaults = the EgoTouch egt_20k/14000 init):
#   TAG (egt14k; exp = <task>_posttrain_<TAG>_realtac_vision8gpu, job name <TAG>_<task>), PRE (merged init dir),
#   HFPREFIX (n0vtla_singlearm_<TAG>_posttrain), NOTE (hf_notes/singlearm_<TAG>_realtac.md),
#   UP_EVERY / UP_FINAL (2000 / 20000 = every 2000 + 19999; UP_EVERY=20000 UP_FINAL=0 = the final step only).
#   TAG=shuf14k PRE=$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_shuf_14000 UP_EVERY=20000 UP_FINAL=0 bash <this> "task1+task3 task2 st" 6 afterok:<smoke>
set -euo pipefail
GROUPS_="${1:?task groups, e.g. \"task1+task3 task2 st\"}"
NSEG="${2:-6}"
FIRST_AFTER="${3:-}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA_egotouch
TAG="${TAG:-egt14k}"
PRE="${PRE:-$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_egotouch_14000}"
HFPREFIX="${HFPREFIX:-n0vtla_singlearm_${TAG}_posttrain}"
NOTE="${NOTE:-deploy/vision/hf_notes/singlearm_${TAG}_realtac.md}"
UP_EVERY="${UP_EVERY:-2000}"; UP_FINAL="${UP_FINAL:-20000}"
[[ -s "$PRE/model.safetensors" ]] || { echo "missing $PRE"; exit 1; }
[[ -f "$NOTE" ]] || { echo "missing $NOTE"; exit 1; }
dep() { [[ "$1" == *:* ]] && echo "$1" || echo "afterany:$1"; }
for group in $GROUPS_; do
  prev="$FIRST_AFTER"
  for t in ${group//+/ }; do
    exp="${t}_posttrain_${TAG}_realtac_vision8gpu"
    for i in $(seq 1 "$NSEG"); do
      args=(--parsable --time=03:00:00 --time-min=01:30:00 --job-name="${TAG}_${t}"
            --export="ALL,TASK=${t},SUFFIX=realtac,PRETRAINED=${PRE},EXP_NAME_OVERRIDE=${exp},STEPS=20000,LRD=20000,SAVE=1000,PRUNE_KEEP=2000,RESUME=auto")
      [[ -n "$prev" ]] && args+=(--dependency="$(dep "$prev")")
      seg=$(sbatch "${args[@]}" deploy/vision/n0vtla_posttrain_predtac_siyuan.sbatch)
      up=$(sbatch --parsable --dependency="afterany:${seg}" \
        --export="ALL,EXP=${exp},PREFIX=${HFPREFIX},EVERY=${UP_EVERY},FINAL=${UP_FINAL},NOTE_FILE=${NOTE}" \
        deploy/vision/n0vtla_upload_posttrain_ckpts_siyuan.sbatch)
      echo "SUB|${exp} seg ${i}: train ${seg} (after ${prev:-none}), upload ${up}"
      prev="$seg"
    done
  done
done
echo "SUB_DONE"
