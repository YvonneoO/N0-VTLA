#!/bin/bash
#SBATCH --job-name=n0vtla_eval_task1_10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_eval_task1_10k_%j.log
#
# Offline eval for both Task1 10k checkpoints (ours + baseline): scripts/eval_wetlab_ship_gate.py,
# the repo's existing "ship gate" -- runs the policy open-loop on genuinely held-out episodes
# (canonical_wetlab_task1_val, never seen in training) and compares the PREDICTED action chunk's
# own step-size/direction-reversal statistics against the wetlab dataset card's measured
# demonstration statistics. Not a task-success eval -- a noise-vs-signal sanity gate (see the
# script's own docstring): a checkpoint whose predicted motion is ~2x demonstration scale or has a
# 40-50% reversal rate is noise-dominated and shouldn't go near the real robot.
#
# The watcher job already deleted both step-10000 checkpoints from VISION right after uploading
# them to HF, so this re-downloads them (whole checkpoint dir -- including optimizer.pt, which
# eval doesn't need, but `hf download --include` only keeps the LAST occurrence when the flag is
# repeated, so a single per-checkpoint ** glob is what's actually reliable) into a separate
# eval_ckpts/ dir, distinct from the training checkpoint_dir.
#
# Submit with: sbatch deploy/vision/n0vtla_eval_task1_10k_ship_gate.sbatch
set -euo pipefail

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
EVAL_CKPT_ROOT="$DATA_ROOT/eval_ckpts/task1_10k"
RESULTS_DIR=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA/eval_results/task1_10k
HF_SRC_ROOT="n0vtla_wetlab_posttrain/post_train_task1_vision8gpu"

mkdir -p "$EVAL_CKPT_ROOT" "$RESULTS_DIR"

echo "=== HF auth check (must be qqyang) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting (env.sh not sourced first?)" >&2
  exit 1
fi

echo
echo "=== downloading step-10000 checkpoints (whole dir -- hf download --include only keeps the LAST" \
     "occurrence when repeated, so a single ** glob is the reliable way to grab the checkpoint) ==="
for name in ours baseline; do
  hf download qqyang/zihiao_real_test --repo-type dataset \
    --include "${HF_SRC_ROOT}/${name}/10000/**" \
    --local-dir "$EVAL_CKPT_ROOT"
done

OURS_CKPT="$EVAL_CKPT_ROOT/${HF_SRC_ROOT}/ours/10000"
BASE_CKPT="$EVAL_CKPT_ROOT/${HF_SRC_ROOT}/baseline/10000"
ls -la "$OURS_CKPT" "$BASE_CKPT"

echo
echo "=== held-out val set ==="
ls "$DATA_ROOT/canonical_wetlab_task1_val/meta/episodes.jsonl"

echo
echo "=== ship-gate eval: ours (step 10000) ==="
python scripts/eval_wetlab_ship_gate.py \
  --config vtla_tactile_posttrain \
  --checkpoint "$OURS_CKPT" \
  --dataset-root "$DATA_ROOT/canonical_wetlab_task1_val" \
  --output "$RESULTS_DIR/ours_10000_ship_gate.json"

echo
echo "=== ship-gate eval: baseline (step 10000) ==="
python scripts/eval_wetlab_ship_gate.py \
  --config vtla_tactile_posttrain \
  --checkpoint "$BASE_CKPT" \
  --dataset-root "$DATA_ROOT/canonical_wetlab_task1_val" \
  --output "$RESULTS_DIR/baseline_10000_ship_gate.json"

echo
echo "=== EVAL_COMPLETE (reports in $RESULTS_DIR) ==="
