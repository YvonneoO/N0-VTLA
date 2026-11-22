#!/bin/bash
#SBATCH --job-name=n0vtla_t4_dl_raw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t4_dl_raw_%j.log
#
# Downloads Task4 (dual-arm 50 mL tube pour, 50 episodes) raw release from the gated repo.
# Full release is 63 GB / 3028 files; this skips what the N0 dual adapter never reads:
# depth_head.mkv (34 GB), imu.txt, the huge config_snapshot *_axes/*_translation json, pico.jsonl.
# Keeps everything the adapter + loader need (H5, 3 RGB videos+csv, both glove npz, robot/{left,right}
# arm_cmd/hand/manifest, operator_note, qc_gate, split/admission json, n0_conversion/).
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
DATA_ROOT=$YQQ/data/n0vtla/wetlab_raw
REL=task4_dual_arm_tube_pour_20261005
mkdir -p "$DATA_ROOT"

WHOAMI=$($HF auth whoami)
echo "hf auth whoami: $WHOAMI"
[[ "$WHOAMI" == *"qqyang"* ]] || { echo "ERROR: expected qqyang" >&2; exit 1; }

$HF download SingleBicycle/tacwam-wetlab-tasks --repo-type dataset --token "$HF_TOKEN" \
  --include "$REL/**" \
  --exclude "*depth_head.mkv" "*imu.txt" "*_axes.json" "*_translation.json" "*pico.jsonl" \
  --local-dir "$DATA_ROOT"

echo "=== verifying ==="
cd "$DATA_ROOT/$REL"
echo "episode dirs: $(ls | grep -cE '^[0-9a-f-]{36}$')  (expect 50)"
echo "episode_30hz.h5: $(ls */episode_30hz.h5 | wc -l)  rgb_head.mp4: $(ls */rgb_head.mp4 | wc -l)  wrist_left.mp4: $(ls */wrist_left.mp4 | wc -l)  wrist_right.mp4: $(ls */wrist_right.mp4 | wc -l)"
echo "left_hand_data.npz: $(ls */left_hand_data.npz | wc -l)  right_hand_data.npz: $(ls */right_hand_data.npz | wc -l)  operator_note.json: $(ls */operator_note.json | wc -l)"
echo "arm_cmd.jsonl: $(ls */robot/left/arm_cmd.jsonl */robot/right/arm_cmd.jsonl | wc -l) (expect 100)"
ls n0_conversion
echo TASK4_RAW_DOWNLOAD_COMPLETE
