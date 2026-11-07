#!/bin/bash
#SBATCH --job-name=n0vtla-midtrain-ts
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --time-min=01:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/midtrain_ts_%j.log
#
# One SEGMENT of a resumable mid-train (Stage-1 online) run on the TouchScale corpus. Submit a chain of these
# with --dependency=afterany:<previous>; every segment resumes from the latest checkpoint of EXP_NAME and a
# segment whose run already reached NUM_TRAIN_STEPS exits at once. --time-min lets Slurm start a segment in a
# backfill gap as short as 1.5 h (gpuboard guide); checkpoints every SAVE_INTERVAL steps bound the loss when a
# segment ends mid-way.
# Required: EXP_NAME, SPLIT_KEY (all | old100). Optional: PAIR_SHUFFLE_SEED (ablation), NUM_TRAIN_STEPS(=20000),
#   SAVE_INTERVAL(=500), NPROC_PER_NODE(=8), BATCH_SIZE (default: config's 64), NUM_WORKERS(=2).
#   sbatch --export=ALL,EXP_NAME=ts_all_20k,SPLIT_KEY=all deploy/vision/n0vtla_midtrain_touchscale_zzh.sbatch
set -uo pipefail
: "${EXP_NAME:?}"; : "${SPLIT_KEY:?all or old100}"
YQQ=/scratch/project/prj-02-phai-lab/yqq
source $YQQ/env.sh
PY=$YQQ/envs/n0vtla/bin/python
CLONE=$YQQ/N0-VTLA_touchscale
IN=$YQQ/data/touchscale_midtrain
cd $CLONE
# The env's n0vtla is an EDITABLE install of the live checkout ($YQQ/N0-VTLA); put this clone first so the
# training imports its own n0vtla package and scripts, not the live checkout's.
export PYTHONPATH=$CLONE:$CLONE/scripts${PYTHONPATH:+:$PYTHONPATH}
$PY -c "import n0vtla, n0vtla.training.itw_online_dataset as d, itw_pressure as ip; print('n0vtla from', n0vtla.__file__); print('itw_pressure from', ip.__file__); assert '$CLONE' in n0vtla.__file__ and hasattr(ip, 'aligned_timeline_h5')" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
export HF_HOME=$YQQ/N0-VTLA/models/huggingface HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export VTLA_ITW_RAW_ROOT=/scratch/project/prj-02-phai-lab/dayou/rss_27/touchscale_data
export VTLA_ITW_EPISODE_LIST_JSON=$IN/manifest.json VTLA_ITW_EPISODE_LIST_KEY=$SPLIT_KEY
export VTLA_ITW_INDEX_CACHE=$IN/index_${SPLIT_KEY}.json
export VTLA_ITW_NORMALIZATION=$IN/normalization_pad30_per_task_scale_touchscale_tacwam.json
export VTLA_PRETRAINED_CHECKPOINT=$YQQ/N0-VTLA/checkpoints/n0-vtla-base
export VTLA_ITW_PAIR_SHUFFLE_SEED="${PAIR_SHUFFLE_SEED:-}"
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
echo "=== $EXP_NAME split=$SPLIT_KEY shuffle_seed=${PAIR_SHUFFLE_SEED:-none} steps=$NUM_TRAIN_STEPS save=$SAVE_INTERVAL gpus=$NPROC limit=$SLURM_JOB_ID ==="
$PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \
  --exp-name="$EXP_NAME" --num-workers="$NUM_WORKERS" \
  --num-train-steps="$NUM_TRAIN_STEPS" --save-interval="$SAVE_INTERVAL" "${EXTRA[@]}"
STATUS=$?
echo "=== segment exit $STATUS; latest checkpoint: $(ls "$CKDIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1) ==="
exit $STATUS
