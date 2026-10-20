#!/bin/bash
#SBATCH --job-name=n0vtla_dl_raw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=prj-02-phai-lab
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/%x_%j.log
#
# Download one wetlab raw release (gated repo SingleBicycle/tacwam-wetlab-tasks; the HF account must be qqyang) into $YQQ/data/n0vtla/wetlab_raw/<REL>.
# Skipped: depth video and imu (not read by the canonical-dataset builder), config_snapshot. About 450 MB per episode.
#   sbatch --export=ALL,REL=task3_whiteboard_erasing_20260918 deploy/vision/n0vtla_wetlab_download_raw_phailab.sbatch
set -uo pipefail
YQQ="${YQQ:-/scratch/project/prj-02-phai-lab/yqq}"
REL="${REL:-task3_whiteboard_erasing_20260918}"
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
$HF auth whoami | grep -q qqyang || { echo "expected the qqyang HF account (gated repo access)" >&2; exit 1; }
RAW=$YQQ/data/n0vtla/wetlab_raw
mkdir -p $RAW
echo "=== $REL -> $RAW"
$HF download SingleBicycle/tacwam-wetlab-tasks --repo-type dataset --token "$HF_TOKEN" --local-dir $RAW \
  --include "$REL/*" --exclude "$REL/*/depth_head.mkv" "$REL/*/depth_head.csv" "$REL/*/imu.txt" "$REL/*/config_snapshot/*" || { echo "DOWNLOAD FAILED"; exit 1; }
echo "episode dirs: $(ls $RAW/$REL | grep -cE '^[0-9a-f-]{36}$')  with robot/manifest.json: $(ls $RAW/$REL/*/robot/manifest.json 2>/dev/null | wc -l)  with wrist_right.mp4: $(ls $RAW/$REL/*/wrist_right.mp4 2>/dev/null | wc -l)  with left_hand_data.npz: $(ls $RAW/$REL/*/left_hand_data.npz 2>/dev/null | wc -l)"
du -sh $RAW/$REL 2>/dev/null | tail -1
echo WETLAB_DOWNLOAD_DONE
