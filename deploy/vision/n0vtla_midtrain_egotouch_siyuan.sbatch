#!/bin/bash
#SBATCH --job-name=n0vtla-midtrain-egt
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --time-min=01:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/midtrain_egt_%j.log
#
# One SEGMENT of a resumable mid-train (Stage-1 online) run on EgoTouch (3-view@224 HDF5, siyuan account;
# loader n0vtla/training/egotouch_online_dataset.py, inputs from n0vtla_egotouch_inputs_siyuan.sbatch). Submit a chain of these
# with --dependency=afterany:<previous>; every segment resumes from the latest checkpoint of EXP_NAME and a
# segment whose run already reached NUM_TRAIN_STEPS exits at once. --time-min lets Slurm start a segment in a
# backfill gap as short as 1.5 h (gpuboard guide); checkpoints every SAVE_INTERVAL steps bound the loss when a
# segment ends mid-way.
# Required: EXP_NAME. Optional: SPLIT_KEY(=train_val), NUM_TRAIN_STEPS(=20000), SAVE_INTERVAL(=500),
#   NPROC_PER_NODE(=8), BATCH_SIZE (default: config's 64), NUM_WORKERS(=2), WANDB(=1).
#   sbatch --export=ALL,EXP_NAME=egt_20k deploy/vision/n0vtla_midtrain_egotouch_siyuan.sbatch
# Smoke (2 GPUs, 40 steps): sbatch --gres=gpu:2 --cpus-per-task=4 --mem=128G --time=01:00:00 --time-min=01:00:00 \
#   --export=ALL,EXP_NAME=smoke_egt,NPROC_PER_NODE=2,NUM_TRAIN_STEPS=40,SAVE_INTERVAL=20 deploy/vision/n0vtla_midtrain_egotouch_siyuan.sbatch
set -uo pipefail
: "${EXP_NAME:?}"; SPLIT_KEY="${SPLIT_KEY:-train_val}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
PY=$YQQ/envs/n0vtla/bin/python
CLONE=$YQQ/N0-VTLA_egotouch
# per-frame-JPEG repack (n0vtla_egotouch_jpeg_repack_siyuan.sbatch); the gzip build e2c_dl/EgoTouch_hdf5_3view_224
# + data/egotouch_midtrain also work, ~30x slower random frame reads
DATA_ROOT="${DATA_ROOT:-$YQQ/data/EgoTouch_jpeg224}"
IN="${IN:-$YQQ/data/egotouch_midtrain_jpeg}"
cd $CLONE
# The env's n0vtla is an EDITABLE install of the live checkout ($YQQ/N0-VTLA); put this clone first so the
# training imports its own n0vtla package and scripts, not the live checkout's.
export PYTHONPATH=$CLONE:$CLONE/scripts${PYTHONPATH:+:$PYTHONPATH}
$PY -c "import n0vtla, n0vtla.training.egotouch_online_dataset as e, itw_pressure as ip; print('n0vtla from', n0vtla.__file__); print('itw_pressure from', ip.__file__); assert '$CLONE' in n0vtla.__file__ and '$CLONE' in ip.__file__" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
export HF_HOME=$YQQ/N0-VTLA/models/huggingface HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0   # as the siyuan post-train launchers
export VTLA_ITW_RAW_ROOT=$DATA_ROOT
export VTLA_ITW_EPISODE_LIST_JSON=$IN/manifest.json VTLA_ITW_EPISODE_LIST_KEY=$SPLIT_KEY
export VTLA_ITW_INDEX_CACHE=$IN/index_${SPLIT_KEY}.json
unset VTLA_ITW_NORMALIZATION VTLA_ITW_PAIR_SHUFFLE_SEED       # EgoTouch grids are already normalized; no shuffle
export VTLA_PRETRAINED_CHECKPOINT=$YQQ/data/n0vtla/n0-vtla-base
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"; SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
NPROC="${NPROC_PER_NODE:-8}"; NUM_WORKERS="${NUM_WORKERS:-2}"

CKDIR=checkpoints/vtla_stage1_predictor_pretrain/$EXP_NAME
LATEST=$(ls "$CKDIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
EXTRA=()
if [[ -n "$LATEST" ]]; then
  if (( LATEST >= NUM_TRAIN_STEPS )); then echo "$EXP_NAME already at step $LATEST >= $NUM_TRAIN_STEPS; nothing to do"; exit 0; fi
  echo "resuming $EXP_NAME from step $LATEST"; EXTRA+=(--resume)
elif [[ -d "$CKDIR" ]]; then
  echo "$CKDIR exists without a checkpoint (a previous segment died before its first save): starting over"
  rm -rf "$CKDIR"
fi
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA+=(--batch-size="$BATCH_SIZE")
[[ "${WANDB:-1}" == "1" ]] && EXTRA+=(--wandb-enabled)
echo "=== $EXP_NAME egotouch split=$SPLIT_KEY steps=$NUM_TRAIN_STEPS save=$SAVE_INTERVAL gpus=$NPROC limit=$SLURM_JOB_ID ==="
$PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" --num-workers="$NUM_WORKERS" \
  --num-train-steps="$NUM_TRAIN_STEPS" --save-interval="$SAVE_INTERVAL" "${EXTRA[@]}"
STATUS=$?
echo "=== segment exit $STATUS; latest checkpoint: $(ls "$CKDIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1) ==="
exit $STATUS
