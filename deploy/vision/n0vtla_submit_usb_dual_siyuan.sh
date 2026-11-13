#!/bin/bash
# Dual-arm USB plug post-train from a merged mid-train init, real tactile, same recipe as the task4 post92traj40k runs (bs64 on
# 8 GPUs, 40000 steps, cosine over 40000) as short backfill segments: NSEG training segments chained afterany (RESUME=auto;
# a segment after completion exits at once), each followed (afterany) by an incremental HF upload of the final step only to
# qqyang/zihiao_real_test:<HFPREFIX>/<exp>/.
# Runs on the VISION login node (only calls sbatch); $(...) is evaluated by THIS bash on VISION.
#   bash deploy/vision/n0vtla_submit_usb_dual_siyuan.sh [NSEG=7] [FIRST_AFTER=<jobid>|<type:jobid>]
# Env: TAG (tsall20k; exp = usb_posttrain_<TAG>_realtac_post100traj40k_vision8gpu), PRE (merged init dir),
#      HFPREFIX (n0vtla_usb_plug_dual_posttrain), NOTE (hf_notes/usb_<TAG>_realtac.md), STEPS (40000).
set -euo pipefail
NSEG="${1:-7}"
FIRST_AFTER="${2:-}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA_usb
TAG="${TAG:-tsall20k}"
PRE="${PRE:-$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_ts_all_20000}"
HFPREFIX="${HFPREFIX:-n0vtla_usb_plug_dual_posttrain}"
NOTE="${NOTE:-deploy/vision/hf_notes/usb_${TAG}_realtac.md}"
STEPS="${STEPS:-40000}"
[[ -s "$PRE/model.safetensors" || "$FIRST_AFTER" == afterok:* ]] || { echo "missing $PRE"; exit 1; }
[[ -f "$NOTE" ]] || { echo "missing $NOTE"; exit 1; }
dep() { [[ "$1" == *:* ]] && echo "$1" || echo "afterany:$1"; }
exp="usb_posttrain_${TAG}_realtac_post100traj$((STEPS / 1000))k_vision8gpu"
prev="$FIRST_AFTER"
for i in $(seq 1 "$NSEG"); do
  args=(--parsable --time=03:00:00 --time-min=01:30:00 --job-name="${TAG}_usb"
        --export="ALL,TASK=usb,DUALSET=post100traj,SUFFIX=realtac,PRETRAINED=${PRE},EXP_NAME_OVERRIDE=${exp},STEPS=${STEPS},LRD=${STEPS},SAVE=1000,PRUNE_KEEP=5000,RESUME=auto")
  [[ -n "$prev" ]] && args+=(--dependency="$(dep "$prev")")
  seg=$(sbatch "${args[@]}" deploy/vision/n0vtla_posttrain_predtac_siyuan.sbatch)
  up=$(sbatch --parsable --dependency="afterany:${seg}" \
    --export="ALL,EXP=${exp},PREFIX=${HFPREFIX},EVERY=${STEPS},FINAL=0,NOTE_FILE=${NOTE}" \
    deploy/vision/n0vtla_upload_posttrain_ckpts_siyuan.sbatch)
  echo "SUB|${exp} seg ${i}: train ${seg} (after ${prev:-none}), upload ${up}"
  prev="$seg"
done
echo "SUB_DONE"
