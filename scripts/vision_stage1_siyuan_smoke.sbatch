#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-siyuan-smoke
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA/stage1_siyuan_smoke_%j.log
set -euo pipefail
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
NORM_PATH=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json
if [ ! -f "$NORM_PATH" ]; then
  hf download qqyang/zihiao_real_test --repo-type dataset --include n0vtla_itw_normalization/normalization_pad30_per_task_scale.json --local-dir /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/norm_dl
  cp /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/norm_dl/n0vtla_itw_normalization/normalization_pad30_per_task_scale.json "$NORM_PATH"
fi
ls -la "$NORM_PATH"
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw
export VTLA_ITW_NORMALIZATION="$NORM_PATH"
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_stage1_online.py vtla_stage1_predictor_pretrain --exp-name=stage1_siyuan_smoke --num-workers=2 --num-train-steps=10 --wandb-enabled
echo VISION_STAGE1_SIYUAN_SMOKE_COMPLETE
