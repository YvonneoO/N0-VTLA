#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-2g
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_2gpu_%j.log
#
# Throughput experiment: 2 GPUs but DELIBERATELY exceeding the usual 2-CPU/GPU ratio
# (8 CPUs for 2 GPUs = 4 CPU/GPU) per user instruction, since decode is confirmed the
# bottleneck for this online loader (single-GPU real-batch run: ~9.5s/step with only 2
# dataloader workers total, vs. a frozen-VLM forward/backward that should be much
# faster) -- more workers should relieve that CPU-bound wait, not GPU compute. 4
# workers/rank x 2 ranks = 8 total, matching the 8-CPU budget.
#
# save_interval dropped to 500 (from the TrainConfig default 2000): a 4h walltime
# window at an ESTIMATED (unverified for this config) ~3s/step would only cross one or
# two 2000-step boundaries, so a link that dies near the end of its window could lose
# most of its progress; 500 gives several checkpoints per window instead.
#
# Chain the same way as vision_train_stage1_online_full.sbatch:
#   First run:  sbatch --export=ALL,EXP_NAME=<name> scripts/vision_train_stage1_online_2gpu_extracpu.sbatch
#   Next link:  sbatch --dependency=afterany:<prev_jobid> \
#                 --export=ALL,EXP_NAME=<same name>,RESUME=1 \
#                 scripts/vision_train_stage1_online_2gpu_extracpu.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="${EXP_NAME:-online_2gpu_${SLURM_JOB_ID}}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers=4 \
  --save-interval=500 \
  --wandb-enabled \
  "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo VISION_STAGE1_ONLINE_2GPU_LAUNCHED
