#!/bin/bash
#SBATCH --job-name=n0vtla_up_t4_norm
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/t4_normup_%j.log
# Uploads the Task4 per-hand tactile normalization alone (same destination as the checkpoint watcher).
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
WHOAMI=$($HF auth whoami)
echo "hf auth whoami: $WHOAMI"
[[ "$WHOAMI" == *"qqyang"* ]] || { echo "ERROR: expected qqyang" >&2; exit 1; }
$HF upload qqyang/zihiao_real_test $YQQ/data/n0vtla/wetlab_tactile_norm_task4_dual.json \
  n0vtla_wetlab_posttrain/post_train_task4_dual_vision8gpu/wetlab_tactile_norm_task4_dual.json \
  --repo-type dataset --token "$HF_TOKEN"
echo T4_NORM_UPLOAD_DONE
