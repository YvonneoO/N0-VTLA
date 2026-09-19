#!/bin/bash
#SBATCH --job-name=n0vtla-upload-scaling-eval
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/upload_scaling_eval_%j.log
#
# Uploads the Stage-1 scaling-study held-out eval results to
# qqyang/zihiao_real_test:n0vtla_pretrain_scaling_study/held_out_eval_results/ :
#   <label>.json        raw per-checkpoint reports (data/stage1_scaling_held_out/)
#   summary.csv         one row per checkpoint, built here from the JSONs
#   logs/eval_<jobid>.log  the eval sbatch logs (per-batch progress + final report)
# Idempotent: re-running just overwrites the same paths, so it is safe both as a one-off
# and as a dependency-triggered watcher (--dependency=afterany:<80pct eval job>).
#
# EVAL_JOBIDS = space-separated eval job ids whose logs to include.
#   sbatch --export=ALL,EVAL_JOBIDS="562817 562818 562819 562820" \
#       deploy/vision/n0vtla_upload_stage1_scaling_eval_results_siyuan.sbatch
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
RES_DIR="${RES_DIR:-$YQQ/data/stage1_scaling_held_out}"
STAGE="$YQQ/data/_stage_scaling_eval_upload"
HF_DEST="${HF_DEST:-n0vtla_pretrain_scaling_study/held_out_eval_results/v1_batchnce_recon}"
: "${EVAL_JOBIDS:?EVAL_JOBIDS is required, e.g. \"562817 562818\"}"

HF=$YQQ/envs/n0vtla/bin/hf
WHOAMI=$($HF auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI'" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/logs"
cp "$RES_DIR"/*.json "$STAGE"/

for jid in $EVAL_JOBIDS; do
  log="$YQQ/stage1_scaling_eval_${jid}.log"
  if [[ -f "$log" ]]; then
    cp "$log" "$STAGE/logs/eval_${jid}.log"
  else
    echo "WARN: $log not found" >&2
  fi
done

$YQQ/envs/n0vtla/bin/python - "$STAGE" <<'PY'
import csv, json, sys
from pathlib import Path

stage = Path(sys.argv[1])
rows = sorted((json.loads(p.read_text()) for p in stage.glob("*.json")), key=lambda r: r["data_fraction"])
cols = ["checkpoint_label", "data_fraction", "checkpoint_step", "num_held_out_episodes",
        "num_batches_scored", "total_valid_samples",
        "mean_stage1_nce", "mean_stage1_recon", "mean_stage1_total"]
with open(stage / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
print((stage / "summary.csv").read_text())
PY

$HF upload qqyang/zihiao_real_test "$STAGE" "$HF_DEST" --repo-type dataset --token "$HF_TOKEN"
echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/${HF_DEST}"

rm -rf "$STAGE"
echo VISION_UPLOAD_SCALING_EVAL_DONE
