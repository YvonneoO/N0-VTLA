#!/bin/bash
#SBATCH --job-name=n0vtla-stage2-smoke
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=150G
#SBATCH --time=00:40:00
#SBATCH --output=stage2_smoke_%j.log
#
# Single-GPU short smoke test of scripts/train_stage2_align_expert.py -- confirms shape/
# freeze/checkpoint correctness on the tiny OpenNeoData slice from
# vision_prep_stage2_smoke_data.sbatch, mirroring exactly how Stage 1's fresh->resume smoke
# chain was validated. --mem=150G reuses the value already validated safe for a single-GPU
# full-corpus Stage-1 smoke test this session (see vision-mem-coupled-to-cpu-oom in
# @TAMU/CLAUDE.md).
#
# Run from inside the isolated Stage-2 git worktree (relative --output path, relative ./assets
# lookup for norm_stats -- see vision_prep_stage2_smoke_data.sbatch's own note on this).
#
#   First run:  sbatch --export=ALL,EXP_NAME=<name> scripts/vision_smoke_stage2.sbatch
#   Resume:     sbatch --dependency=afterany:<prev_jobid> \
#                 --export=ALL,EXP_NAME=<same name>,RESUME=1,NUM_TRAIN_STEPS=10 \
#                 scripts/vision_smoke_stage2.sbatch
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base
export VTLA_STAGE1_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/vtla_stage1_predictor_pretrain/online_full_v3
export VTLA_DATASET_PATH=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/data/openneodata_flexiv_smoke
export VTLA_ASSET_ID=openneodata_flexiv_smoke

EXP_NAME="${EXP_NAME:-stage2_smoke}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_stage2_align_expert.py vtla_stage2_align_expert \
  --exp-name="$EXP_NAME" \
  --num-workers=2 \
  --batch-size=2 \
  --save-interval="$SAVE_INTERVAL" \
  --num-train-steps="$NUM_TRAIN_STEPS" \
  "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo STAGE2_SMOKE_COMPLETE
