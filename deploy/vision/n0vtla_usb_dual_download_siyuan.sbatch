#!/bin/bash
#SBATCH --job-name=n0vtla_usb_dl
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_usb_dl_%j.log
#
# Dual-arm USB plug (T10) teleop release, both batches, from the gated SingleBicycle/rss-dual-arm-tactile repo:
#   task_usb_plug_dual_20261108 (batch 1, 57 successes, no pushes) + task_usb_plug_dual_20261110 (batch 2, 43 successes,
#   teleop pushes on) = the operator's 100 train trajectories (no val/test split: the physical rollout is the evaluation).
# Full size 174 GB; skips what the dual adapter never reads (same excludes as n0vtla_task4_download_raw.sbatch):
# depth_head.mkv, imu.txt, the config_snapshot *_axes/*_translation json, pico.jsonl -> ~96 GB.
# Not downloaded: task_usb_plug_dual_20261108_failures (failed/aborted attempts, not training data).
# Gated wetlab data: stays on VISION, never re-uploaded to a public repo.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
DATA_ROOT=$YQQ/data/n0vtla/wetlab_raw
mkdir -p "$DATA_ROOT"
WHOAMI=$($HF auth whoami)   # HF_TOKEN (env.sh) takes precedence over the account's cached login
echo "hf auth whoami: $WHOAMI"
[[ "$WHOAMI" == *"qqyang"* ]] || { echo "ERROR: expected qqyang" >&2; exit 1; }
export HF_HUB_DISABLE_XET=1   # stalls seen with xet on large gated pulls
# ~6400 files: HF rate-limits the token (429 Too Many Requests, seen 2026-11-13 with the default 8 workers), so few workers
# and retries with a pause; hf download resumes (files already complete are skipped).
for REL in task_usb_plug_dual_20261108 task_usb_plug_dual_20261110; do
  for try in 1 2 3 4 5 6 7 8; do
    echo "=== $REL (attempt $try)"
    if $HF download SingleBicycle/rss-dual-arm-tactile --repo-type dataset --token "$HF_TOKEN" --max-workers 2 \
         --include "$REL/**" \
         --exclude "*depth_head.mkv" "*imu.txt" "*_axes.json" "*_translation.json" "*pico.jsonl" \
         --local-dir "$DATA_ROOT" > "$DATA_ROOT/.usb_dl_$REL.log" 2>&1; then
      echo "$REL complete"; break
    fi
    tail -2 "$DATA_ROOT/.usb_dl_$REL.log"
    [ "$try" = 8 ] && { echo "FATAL: $REL still failing after 8 attempts"; exit 1; }
    sleep 600
  done
done
echo "=== verifying"
for REL in task_usb_plug_dual_20261108 task_usb_plug_dual_20261110; do
  cd "$DATA_ROOT/$REL"
  echo "$REL: episode dirs $(ls | grep -cE '^[0-9a-f-]{36}$')  train $($YQQ/envs/n0vtla/bin/python -c "import json;print(len(json.load(open('training_split.json'))['train']))")"
  echo "  h5 $(ls */episode_30hz.h5 | wc -l)  rgb_head $(ls */rgb_head.mp4 | wc -l)  wrist_left $(ls */wrist_left.mp4 | wc -l)  wrist_right $(ls */wrist_right.mp4 | wc -l)"
  echo "  left_npz $(ls */left_hand_data.npz | wc -l)  right_npz $(ls */right_hand_data.npz | wc -l)  operator_note $(ls */operator_note.json | wc -l)  arm_cmd $(ls */robot/left/arm_cmd.jsonl */robot/right/arm_cmd.jsonl | wc -l)"
done
echo USB_RAW_DOWNLOAD_COMPLETE
