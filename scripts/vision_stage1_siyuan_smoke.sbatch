#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-siyuan-smoke
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/stage1_siyuan_smoke_%j.log
#
# Stage-1 online-loader smoke test on the siyuan account (prj-02-uq-llms-for-reasoning).
# Only needs to prove the env + code path work end-to-end -- 2 GPUs is enough for that
# (a smoke test does not need a full 8-GPU node; --gres=gpu:8 just made this queue
# behind other jobs for a whole free node instead of scheduling fast on partial
# capacity). For the REAL 20/40/60/80% scaling runs later, use a dedicated launcher
# with --gres=gpu:8 (see vision_train_stage1_online_full_siyuan.sbatch).
#
# --time=00:45:00 (not 20): job 560914's attempt showed setup alone (env activate +
# imports + distributed init ~11.3min, checkpoint load ~1.4min, dataset build +
# dataloader worker spawn ~5.3min before it crashed on missing ffmpeg) consumed
# ~18 of its 20 minutes WITHOUT completing a single training step -- see project
# memory vision-tolerant-walltime (timeouts lose everything, can't be extended).
# Scoped to the 20% manifest (not the full 39426-episode raw root) so this smoke
# test's dataset-build cost matches what the real 20%-data run will actually see.
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

NORM_PATH=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json
if [[ ! -f "$NORM_PATH" ]]; then
  mkdir -p "$(dirname "$NORM_PATH")"
  hf download qqyang/zihiao_real_test \
    n0vtla_itw_normalization/normalization_pad30_per_task_scale.json \
    --repo-type dataset --token "$HF_TOKEN" \
    --local-dir /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_norm_download
  mv /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_norm_download/n0vtla_itw_normalization/normalization_pad30_per_task_scale.json "$NORM_PATH"
  rm -rf /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_norm_download
fi

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw
export VTLA_ITW_NORMALIZATION="$NORM_PATH"
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base
export VTLA_ITW_EPISODE_LIST_JSON=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json
export VTLA_ITW_EPISODE_LIST_KEY=20pct

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name=stage1_siyuan_smoke \
  --num-workers=2 \
  --num-train-steps=10 \
  --wandb-enabled

echo VISION_STAGE1_ONLINE_SIYUAN_SMOKE_DONE
