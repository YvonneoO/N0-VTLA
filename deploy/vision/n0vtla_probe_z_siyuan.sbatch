#!/bin/bash
#SBATCH --job-name=n0vtla-probe-z
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_probe_z_%x_%j.log
#
# Tactile causal probe (scripts/probe_z_tactile_dependence.py, docs/TACTILE_CAUSAL_PROBE.md): does the tactile latent z of a post-train checkpoint
# actually read the tactile input?  R = tactile sensitivity / vision-language sensitivity of z (centered): R >= 3 healthy, 0.3-3 mixed, < 0.3 VL-router.
# Default = the REAL-tactile post-train checkpoint of TASK (touchscale project, 10000 steps) on its held-out episodes; override CKPT / DATA / ASSET / LABEL
# to probe another checkpoint (e.g. the predicted-tactile run). Results: data/predtac_offline_check/probe_<LABEL>.{json,md}
#   sbatch --job-name=task1 --export=ALL,TASK=task1 deploy/vision/n0vtla_probe_z_siyuan.sbatch      (TASK task1 | sv2)
set -uo pipefail
: "${TASK:?TASK=task1|sv2}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
source activate $YQQ/envs/n0vtla
PY=$YQQ/envs/n0vtla/bin/python
export PATH="$YQQ/envs/n0vtla/bin:$PATH"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HOME=$YQQ/N0-VTLA/models/huggingface
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export VTLA_ATTN_IMPL=eager VTLA_PREFIX_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=$YQQ/data/n0vtla
REF=$DATA/eval_ckpts
case "$TASK" in
  task1) C="$REF/task1_10k/n0vtla_wetlab_posttrain/post_train_task1_vision8gpu/ours/10000"; D=$DATA/canonical_wetlab_task1_val; A=task1_tuberack_train;;
  sv2)   C="$REF/sv2_100pct/n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_10000"; D=$DATA/canonical_wetlab_smoketestv2_dev; A=wetlab_smoketestv2_train;;
  *) echo "unknown TASK $TASK"; exit 2;;
esac
CKPT="${CKPT:-$C/model.safetensors}"
export VTLA_DATASET_PATH="${DATA_PATH:-$D}" VTLA_ASSET_ID="${ASSET:-$A}"
LABEL="${LABEL:-${TASK}_realtactile}"
OUT=$YQQ/data/predtac_offline_check; mkdir -p $OUT
[ -f "$CKPT" ] || { echo "FATAL: checkpoint missing $CKPT"; exit 2; }
echo "=== probe $LABEL: ckpt=$CKPT data=$VTLA_DATASET_PATH asset=$VTLA_ASSET_ID ==="
$PY scripts/probe_z_tactile_dependence.py --config vtla_tactile_posttrain --ckpt "$CKPT" --batches 16 --batch-size 16 \
  --out-json $OUT/probe_${LABEL}.json --out-md $OUT/probe_${LABEL}.md --markdown
echo PROBE_DONE
