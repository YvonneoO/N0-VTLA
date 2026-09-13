#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-online-full
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/stage1_online_full_%j.log
#
# Formal (not exploratory) online Stage-1 training run: full itw07-23..itw09-10
# corpus (no date/episode cap), real batch_size=64 and num_train_steps=20_000 (the
# vtla_stage1_predictor_pretrain TrainConfig's own defaults -- neither overridden
# here), the new full-corpus per-task-scale table, single-node DDP. Gated on job
# 535038 (single-GPU, itw08-10-only, real-batch-size convergence check) showing a
# clean decreasing nce trend with no instability through step 150.
#
# 2026-09-13: `sbatch --test-only` showed the SAME ~2026-09-14T18:2x estimated start
# regardless of GPU count (1/2/4/8) or walltime (4h/24h) -- this account's queue is
# priority/fairshare-bound right now (a 4-node job + several pending jobs ahead), not
# resource-fit-bound, so shrinking the request doesn't change the current snapshot
# estimate. Still shrunk to 4 GPUs + 4h (smaller fairshare footprint, more backfill
# opportunities as the queue churns) and split into a --resume chain instead of one
# long request, per user instruction, rather than holding out for one 24h slot.
#
# Chaining: NPROC_PER_NODE/EXP_NAME/RESUME are all overridable via --export=ALL,...
#   First run:  sbatch --export=ALL,EXP_NAME=<name> scripts/vision_train_stage1_online_full.sbatch
#   Next link:  sbatch --dependency=afterany:<prev_jobid> \
#                 --export=ALL,EXP_NAME=<same name>,RESUME=1 \
#                 scripts/vision_train_stage1_online_full.sbatch
# (checkpoints save every save_interval=2000 steps by TrainConfig default, so a link
# that dies mid-way loses at most the steps since the last checkpoint, not the run.)
#
# 2026-09-13 postmortem: the first attempt (job 535274) was SLURM-OOM-killed ~47s in
# ("Detected 1 oom_kill event in StepId=535274.batch"), well before step 1 or any
# checkpoint -- no --mem was set, and the job's own 4 DDP ranks EACH independently
# build the full ~32,000-episode corpus index in RAM (index_episodes has no rank-0-
# builds-and-broadcasts sharing), so 4x the indexing memory hit whatever small default
# the cgroup allocated for --cpus-per-task=8 (see @TAMU/CLAUDE.md's documented
# vision-mem-coupled-to-cpu-oom pitfall -- same failure mode, worse here from the 4x
# redundant indexing). Its --resume follow-up (535275) then failed instantly for the
# obvious reason: no checkpoint existed yet to resume from. Fix: explicit --mem above.
set -euo pipefail

source /scratch/project/prj-02-phai-lab/yqq/env.sh
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
python -c "import jax, torch; print('env check OK: jax', jax.__version__, 'torch', torch.__version__, 'GPUs', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
# No VTLA_ITW_DATES / VTLA_ITW_MAX_EPISODES -- full corpus, every date under raw_root.
export VTLA_ITW_NORMALIZATION=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/per_task_tactile_scale/normalization_pad30_per_task_scale.json
export VTLA_PRETRAINED_CHECKPOINT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

EXP_NAME="${EXP_NAME:-online_full_${SLURM_JOB_ID}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" \
  --num-workers=2 \
  --wandb-enabled \
  "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo VISION_STAGE1_ONLINE_FULL_LAUNCHED
