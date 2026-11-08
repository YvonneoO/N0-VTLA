#!/bin/bash
#SBATCH --job-name=n0vtla_predtac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_predtac_%x_%j.log
#
# N0-VTLA post-train with PREDICTED tactile (video -> fine-tuned v2 DiT -> cell-to-taxel decoder -> glove-format tactile -> canonical dataset,
# see Ego2Contact scripts/tools/robot_ft_predtac_build.sbatch). Same recipe as task1_posttrain_ours_vision8gpu / sv2 (bs64 on 8 GPUs, config
# default lr schedule, init = Stage-1 human-data mid-train merged into the base) -- only the tactile VIDEOS of the dataset differ (state, action,
# RGB, episodes, norm_stats are identical to the real-tactile datasets, so the same asset id is used).
#   TASK   = task1 | sv2 | task3
#   SUFFIX = predtac (dataset suffix written by robot_ft_predtac_build.sbatch)
#   STEPS  = 20000 (smoke: 20)     SAVE = 5000      RESUME=1 to resume, RESUME=auto to resume only if a checkpoint exists (else start fresh)
#   sbatch --job-name=task1 --export=ALL,TASK=task1 deploy/vision/n0vtla_posttrain_predtac_siyuan.sbatch
#   BS = global batch (default 64; a 1-GPU smoke uses 8 = the per-GPU batch of the 8-GPU run)
#   Task4 dual, EgoTouch mid-train init (same recipe as the post92traj40k ours runs):
#   --export=ALL,TASK=task4,DUALSET=post92traj,SUFFIX=realtac,PRETRAINED=$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_egotouch_20000,
#     EXP_NAME_OVERRIDE=task4_posttrain_egt14k_realtac_post92traj40k_vision8gpu,STEPS=40000,LRD=40000,SAVE=1000,PRUNE_KEEP=5000,RESUME=auto
#   with short backfill segments: sbatch --time=03:00:00 --time-min=01:30:00 ... (chain with --dependency=afterany)
set -uo pipefail
: "${TASK:?TASK must be task1, sv2, task3 or task4}"
SUFFIX="${SUFFIX:-predtac}"
YQQ="${YQQ:-/scratch/project/prj-02-uq-llms-for-reasoning/yqq}"   # other account: --export=ALL,YQQ=<its yqq> --account=<its account> --output=<its logs>
cd $YQQ/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
export CONDA_PREFIX="$YQQ/envs/n0vtla"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$YQQ/envs/n0vtla/bin:$PATH"

DATA_ROOT=$YQQ/data/n0vtla
case "$TASK" in
  task1) export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task1_${SUFFIX}_train"; export VTLA_ASSET_ID=task1_tuberack_train ;;
  sv2)   export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_smoketestv2_${SUFFIX}_train"; export VTLA_ASSET_ID=wetlab_smoketestv2_train ;;
  task3) export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task3_${SUFFIX}_train"; export VTLA_ASSET_ID=task3_whiteboard_train ;;
  task4) DSET="${DUALSET:-post42}"   # post42 = the 42 batch-1 post-train episodes; post92 = those + the 50 batch-2 episodes
         export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task4_dual_${DSET}_${SUFFIX}"; export VTLA_ASSET_ID=task4_dual_pour_${DSET} ;;   # dual-arm, 42 post-train episodes (dual42 plan)
  *) echo "unknown TASK=$TASK" >&2; exit 1 ;;
esac
[ -d "$VTLA_DATASET_PATH" ] || { echo "dataset $VTLA_DATASET_PATH missing"; exit 2; }
export VTLA_PRETRAINED_CHECKPOINT="${PRETRAINED:-$DATA_ROOT/n0-vtla-base_plus_stage1_human_14000}"
export EXP_NAME="${EXP_NAME_OVERRIDE:-${TASK}_posttrain_ours_${SUFFIX}_vision8gpu}"
export NPROC_PER_NODE=${GPUS:-8}
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"
ulimit -n 65536 2>/dev/null || true

$YQQ/envs/n0vtla/bin/python -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('facebook/dinov2-base')
print('DINOv2 config cached OK')
"

echo "=== $EXP_NAME: ckpt=$VTLA_PRETRAINED_CHECKPOINT data=$VTLA_DATASET_PATH asset=$VTLA_ASSET_ID steps=${STEPS:-20000} ==="
# RESUME=1: force --resume.  RESUME=auto: resume only if the experiment dir holds a saved step (numeric dir, saves are atomic), else start from step 0
# (a leftover dir without any checkpoint is overwritten).
RFLAG=""
CKD="$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain/$EXP_NAME"
if [ "${RESUME:-}" = "auto" ]; then
  if ls -d "$CKD"/[0-9]* >/dev/null 2>&1; then
    LATEST=$(ls -d "$CKD"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
    if [ "$LATEST" -ge "${STEPS:-20000}" ]; then echo "RESUME=auto: latest step $LATEST >= STEPS ${STEPS:-20000}, training already complete -> nothing to do"; echo "=== TRAINING SCRIPT EXITED ($EXP_NAME) rc=0 (already complete) ==="; exit 0; fi
    echo "RESUME=auto: checkpoint found in $CKD (latest step $LATEST) -> --resume"; RFLAG="--resume"
  else echo "RESUME=auto: no checkpoint in $CKD -> fresh start from step 0"; [ -d "$CKD" ] && RFLAG="--overwrite"; fi
elif [ -n "${RESUME:-}" ]; then RFLAG="--resume"; fi
# LRD: cosine decay length (config default 20000). For a run longer than 20k steps (segmented/resumed) EVERY segment must use the same LRD, else the LR
# would sit flat at decay_lr after step 20000 or jump between segments.
LRFLAG=""; [ -n "${LRD:-}" ] && LRFLAG="--lr-schedule.decay-steps=$LRD"
# PRUNE_KEEP=N (backfill-friendly short segments with a small SAVE): train_pytorch keeps every save (~22 GB each), so
# before, during (every 10 min) and after training, every step older than the latest is pruned: multiples of N and
# STEPS-1 (the eval/upload points, as the 5000-step runs keep) lose only optimizer.pt, all other steps are removed.
# The latest complete step (saves are atomic: tmp_<step> renamed) is always kept whole for --resume.
prune() {
  [ -n "${PRUNE_KEEP:-}" ] && [ -d "$CKD" ] || return 0
  local latest d st
  latest=$(ls -d "$CKD"/[0-9]* 2>/dev/null | xargs -r -n1 basename | sort -n | tail -1)
  [ -n "$latest" ] || return 0
  for d in "$CKD"/[0-9]*; do
    st=$(basename "$d"); [ "$st" = "$latest" ] && continue
    if (( st % PRUNE_KEEP == 0 || st == ${STEPS:-20000} - 1 )); then rm -f "$d/optimizer.pt"; else rm -rf "$d"; echo "PRUNE|removed step $st"; fi
  done
}
prune
PRUNER=""
if [ -n "${PRUNE_KEEP:-}" ]; then ( while sleep 600; do prune; done ) & PRUNER=$!; fi
bash train.sh --batch-size=${BS:-64} --num-train-steps=${STEPS:-20000} --save-interval=${SAVE:-5000} $LRFLAG $RFLAG
RC=$?
[ -n "$PRUNER" ] && kill "$PRUNER" 2>/dev/null
prune
echo "=== TRAINING SCRIPT EXITED ($EXP_NAME) rc=$RC ==="
exit $RC
