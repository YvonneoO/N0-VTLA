#!/bin/bash
#SBATCH --job-name=n0vtla_up_usb_norm
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/usb_normup_%j.log
# Uploads the USB plug dual-arm normalizations the policy needs at serving time (the client must use the SAME tactile norm as
# training), next to the post-train runs in qqyang/zihiao_real_test:n0vtla_usb_plug_dual_posttrain/:
#   wetlab_tactile_norm_usb_dual.json   per-hand pad30 tactile norm (fit_wetlab_dual_tactile_norm.py on the 100 train trajectories)
#   norm_stats_usb_plug_dual_post100traj.json   state/action norm stats (assets/vtla_tactile_posttrain/usb_plug_dual_post100traj)
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
WHOAMI=$($HF auth whoami)
echo "hf auth whoami: $WHOAMI"
[[ "$WHOAMI" == *"qqyang"* ]] || { echo "ERROR: expected qqyang" >&2; exit 1; }
P=n0vtla_usb_plug_dual_posttrain
$HF upload qqyang/zihiao_real_test $YQQ/data/n0vtla/wetlab_tactile_norm_usb_dual.json $P/wetlab_tactile_norm_usb_dual.json \
  --repo-type dataset --token "$HF_TOKEN"
$HF upload qqyang/zihiao_real_test $YQQ/N0-VTLA/assets/vtla_tactile_posttrain/usb_plug_dual_post100traj/norm_stats.json \
  $P/norm_stats_usb_plug_dual_post100traj.json --repo-type dataset --token "$HF_TOKEN"
echo USB_NORM_UPLOAD_DONE
