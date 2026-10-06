#!/bin/bash
#SBATCH --job-name=n0vtla-t4-replay
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/t4_replay_%j.log
#
# Replay check of a Task4 dual-arm checkpoint through the real inference path
# (scripts/check_dual_policy_replay.py). Required: CKPT (dir with model.safetensors + assets/).
# Optional: LABEL (output name), NUM_SAMPLES(=24).
#   sbatch --export=ALL,CKPT=<ckpt dir>,LABEL=ours_2000 deploy/vision/n0vtla_check_dual_replay_siyuan.sbatch
set -euo pipefail
: "${CKPT:?CKPT is required}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"
cd $YQQ/N0-VTLA
export PYTHONPATH=$PWD:$PWD/scripts
LABEL="${LABEL:-replay}"
$YQQ/envs/n0vtla/bin/python scripts/check_dual_policy_replay.py \
  --checkpoint "$CKPT" --dataset $YQQ/data/n0vtla/canonical_wetlab_task4_dual_train \
  --asset-id task4_dual_pour_train --num-samples "${NUM_SAMPLES:-24}" \
  --output $YQQ/data/n0vtla/task4_replay_${LABEL}.json
echo T4_REPLAY_DONE
