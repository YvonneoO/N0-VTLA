#!/bin/bash
#SBATCH --job-name=n0vtla_data_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_data_download_%j.log

set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq

module load Miniforge3/25.3.0-3
source env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
mkdir -p "$DATA_ROOT"

echo "=== [1/4] downloading pretrained checkpoint (n0-vtla-base_plus_stage1_human_14000, ~8.25GB) ==="
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0-vtla-base_plus_stage1_human_14000/**" \
  --local-dir "$DATA_ROOT"

echo "=== [2/4] downloading canonical_wetlab_task1_train ==="
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "canonical_wetlab_task1_train/**" \
  --local-dir "$DATA_ROOT"

echo "=== [3/4] downloading canonical_wetlab_task1_val ==="
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "canonical_wetlab_task1_val/**" \
  --local-dir "$DATA_ROOT"

echo "=== [4/4] downloading norm_stats.json and placing at assets/vtla_tactile_posttrain/task1_tuberack_train/ ==="
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "task1_tuberack_train_norm_stats.json" \
  --local-dir "$DATA_ROOT"

N0VTLA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA
mkdir -p "$N0VTLA_ROOT/assets/vtla_tactile_posttrain/task1_tuberack_train"
cp "$DATA_ROOT/task1_tuberack_train_norm_stats.json" \
   "$N0VTLA_ROOT/assets/vtla_tactile_posttrain/task1_tuberack_train/norm_stats.json"

echo "=== verifying ==="
ls -la "$DATA_ROOT/n0-vtla-base_plus_stage1_human_14000/"
du -sh "$DATA_ROOT/canonical_wetlab_task1_train" "$DATA_ROOT/canonical_wetlab_task1_val"
ls -la "$N0VTLA_ROOT/assets/vtla_tactile_posttrain/task1_tuberack_train/norm_stats.json"
echo "=== DOWNLOAD DONE ==="
