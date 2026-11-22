#!/bin/bash
#SBATCH --job-name=n0vtla-t4-sens
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/t4_sens_%j.log
# Sampling-spread + input-sensitivity probe of a Task4 checkpoint (scripts/check_dual_policy_sensitivity.py).
# Required: CKPT. Optional: LABEL, NUM_FRAMES(=48), K(=12).
#   sbatch --export=ALL,CKPT=<ckpt dir>,LABEL=ours_20000 deploy/vision/n0vtla_check_dual_sensitivity_siyuan.sbatch
set -euo pipefail
: "${CKPT:?CKPT is required}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"
cd $YQQ/N0-VTLA
export PYTHONPATH=$PWD:$PWD/scripts
$YQQ/envs/n0vtla/bin/python scripts/check_dual_policy_sensitivity.py --checkpoint "$CKPT" \
  --dataset $YQQ/data/n0vtla/canonical_wetlab_task4_dual_train --asset-id task4_dual_pour_train \
  --num-frames "${NUM_FRAMES:-48}" --k-samples "${K:-12}" --output $YQQ/data/n0vtla/task4_sens_${LABEL:-x}.json
echo T4_SENS_DONE
