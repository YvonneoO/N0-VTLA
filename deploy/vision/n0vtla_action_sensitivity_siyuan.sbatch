#!/bin/bash
#SBATCH --job-name=n0vtla-act-sens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_act_sens_%x_%j.log
#
# Action sensitivity to the tactile input: the REAL-tactile post-train checkpoint of TASK samples action chunks (same noise per row) on the same
# held-out episodes fed with real / predicted / all-zero tactile; the chunks are compared in physical units, overall and on contact rows
# (scripts/dump_posttrain_sampled_actions.py + scripts/compare_tactile_action_sensitivity.py). Output: data/predtac_offline_check/actsens_<TASK>.{md,json}
#   sbatch --job-name=task1 --export=ALL,TASK=task1 deploy/vision/n0vtla_action_sensitivity_siyuan.sbatch      (TASK task1 | sv2; STRIDE=6, MAXROWS for a smoke)
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
DATA=$YQQ/data/n0vtla; REF=$DATA/eval_ckpts
OUT=$YQQ/data/predtac_offline_check/actsens_$TASK; mkdir -p $OUT
case "$TASK" in
  task1) CKPT="$REF/task1_10k/n0vtla_wetlab_posttrain/post_train_task1_vision8gpu/ours/10000"; ASSET=task1_tuberack_train
         REAL=$DATA/canonical_wetlab_task1_val; PRED=$DATA/canonical_wetlab_task1_predtacq_val; ZERO=$DATA/canonical_wetlab_task1_zerotac0_val;;
  sv2)   CKPT="$REF/sv2_100pct/n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_10000"; ASSET=wetlab_smoketestv2_train
         REAL=$DATA/canonical_wetlab_smoketestv2_dev; PRED=$DATA/canonical_wetlab_smoketestv2_predtacq_val; ZERO=$DATA/canonical_wetlab_smoketestv2_zerotac0_val;;
  *) echo "unknown TASK $TASK"; exit 2;;
esac
for v in real pred zero; do
  case "$v" in real) ds=$REAL;; pred) ds=$PRED;; zero) ds=$ZERO;; esac
  [ -f "$OUT/$v.npz" ] && { echo "SKIP $v (exists)"; continue; }
  echo "=== dump $v ($ds) ==="
  $PY scripts/dump_posttrain_sampled_actions.py --checkpoint "$CKPT" --dataset-root "$ds" --asset-id "$ASSET" --output "$OUT/$v.npz" \
    --stride ${STRIDE:-6} ${MAXROWS:+--max-rows $MAXROWS} || echo "FAILED $v"
done
$PY scripts/compare_tactile_action_sensitivity.py --real $OUT/real.npz --pred $OUT/pred.npz --zero $OUT/zero.npz --label "$TASK: real-tactile post-train checkpoint" \
  --out $OUT/../actsens_$TASK.md
echo ACT_SENS_DONE
