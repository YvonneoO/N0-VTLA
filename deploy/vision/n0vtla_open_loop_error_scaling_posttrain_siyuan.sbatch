#!/bin/bash
#SBATCH --job-name=n0vtla-ol-err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_ol_err_%j.log
#
# Open-loop SAMPLED-action error (mm / hand counts, by horizon) for the scaling-study post-train checkpoints (10000 step):
# sv2 runs on canonical_wetlab_smoketestv2_dev (7 episodes), task1 runs on
# canonical_wetlab_task1_val (13 episodes), each with its own TRAIN asset's normalization.
# One GPU, checkpoints scored one after another; writes <label>.json per checkpoint and
# summary.csv into OUT_DIR. Runs from the live N0-VTLA checkout (same code the runs trained with).
# 0pct/100pct = official-base / Stage-1-human-14000 post-train references (task1: HF post_train_task1_vision8gpu
# baseline/ours 10000; sv2 100%: HF human_data_post_train_v2 10000; sv2 0%: HF checkpoint_20000 = official base
# post-trained 20k steps on the OLD smoke_test data with its OWN norm stats (asset wetlab_v2_train), scored on sv2 dev).
#   sbatch [--export=ALL,STEP=10000,CELLS="sv2:20pct task1:20pct"] deploy/vision/n0vtla_heldout_loss_scaling_posttrain_siyuan.sbatch
set -uo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
source activate $YQQ/envs/n0vtla
# The submitting shell may have another venv active (its PATH is inherited); pin the interpreter.
PY=$YQQ/envs/n0vtla/bin/python
export PATH="$YQQ/envs/n0vtla/bin:$PATH"

export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HOME=$YQQ/N0-VTLA/models/huggingface
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export VTLA_ATTN_IMPL=eager VTLA_PREFIX_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=$YQQ/data/n0vtla
OUT_DIR="${OUT_DIR:-$YQQ/data/scaling_posttrain_open_loop_error}"
STEP="${STEP:-10000}"
CELLS="${CELLS:-sv2:0pct sv2:20pct sv2:40pct sv2:60pct sv2:80pct sv2:100pct task1:0pct task1:20pct task1:40pct task1:60pct task1:80pct task1:100pct}"
REF=$DATA/eval_ckpts
mkdir -p "$OUT_DIR"

for cell in $CELLS; do
  task="${cell%%:*}"; pct="${cell##*:}"
  case "$task" in
    sv2)   ds="$DATA/canonical_wetlab_smoketestv2_dev"; asset=wetlab_smoketestv2_train ;;
    task1) ds="$DATA/canonical_wetlab_task1_val";       asset=task1_tuberack_train ;;
  esac
  label="${task}_${pct}"
  out="$OUT_DIR/${label}.json"
  [[ -f "$out" ]] && { echo "SKIP $label (exists)"; continue; }
  case "$label" in
    task1_0pct)   ckpt="$REF/task1_10k/n0vtla_wetlab_posttrain/post_train_task1_vision8gpu/baseline/${STEP}" ;;
    task1_100pct) ckpt="$REF/task1_10k/n0vtla_wetlab_posttrain/post_train_task1_vision8gpu/ours/${STEP}" ;;
    sv2_100pct)   ckpt="$REF/sv2_100pct/n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_${STEP}" ;;
    sv2_0pct)     ckpt="$REF/sv2_0pct_old/n0vtla_wetlab_posttrain/checkpoint_20000"; asset=wetlab_v2_train ;;
    *)            ckpt="checkpoints/vtla_tactile_posttrain/${task}_posttrain_s1_${pct}_10k/${STEP}" ;;
  esac
  echo "=== $label ($ckpt) ==="
  $PY scripts/compute_posttrain_open_loop_error.py \
    --checkpoint "$ckpt" \
    --checkpoint-label "$label" --dataset-root "$ds" --asset-id "$asset" --output "$out" \
    || echo "FAILED $label"
done

$PY - "$OUT_DIR" <<'PY'
import csv, json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
cols = ["checkpoint_label", "n_episodes", "n_scored_rows", "xyz_mm", "xyz_mm_se", "xyz_mm_zero_motion",
        "hand_mae", "hand_mae_se", "hand_mae_mean_cmd", "rot6d_mae"]
with open(d / "summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in rows:
        m = r["metrics"]
        w.writerow([r["checkpoint_label"], r["n_episodes"], r["n_scored_rows"],
                    m["sample_xyz_mm"]["mean"], m["sample_xyz_mm"]["episode_se"], m["zero_xyz_mm"]["mean"],
                    m["sample_hand_mae"]["mean"], m["sample_hand_mae"]["episode_se"], m["zero_hand_mae"]["mean"],
                    m["sample_rot6d_mae"]["mean"]])
print((d / "summary.csv").read_text())
PYecho OPEN_LOOP_ERR_DONE
