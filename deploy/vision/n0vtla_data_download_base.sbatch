#!/bin/bash
#SBATCH --job-name=n0vtla_dl_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_dl_base_%j.log

set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq

module load Miniforge3/25.3.0-3
source env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla

echo "=== downloading plain n0-vtla-base checkpoint (8.25GB) ==="
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0-vtla-base/**" \
  --local-dir "$DATA_ROOT"

echo "=== verifying ==="
ls -la "$DATA_ROOT/n0-vtla-base/"
echo "=== DOWNLOAD BASE DONE ==="
