#!/bin/bash
#SBATCH --job-name=n0vtla-upload-ptdata
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/upload_ptdata_%j.log
#
# Watcher for the Task1 post-train-data eval (n0vtla_stage1_scaling_posttrain_eval_siyuan.sbatch):
# submit with --dependency=afterany:<eval job ids>. Aggregates whatever checkpoints finished
# (both splits present) into summary.json/csv (per split, train+val pooled, bootstrap CIs, deltas
# vs `base`) and uploads results + embeddings + eval logs to
# qqyang/zihiao_real_test:n0vtla_pretrain_scaling_study/held_out_eval_results/v3_task1ptdata_retr/.
# Idempotent. EVAL_JOBIDS (space-separated) selects the logs to include; LABELS defaults to
# "base 20pct 40pct 60pct 80pct 100pct".
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
cd $YQQ/N0-VTLA_scaling_extract
: "${EVAL_JOBIDS:?}"
RES=$YQQ/data/stage1_scaling_ptdata_eval
STAGE=$YQQ/data/_stage_ptdata_upload
HF_DEST="${HF_DEST:-n0vtla_pretrain_scaling_study/held_out_eval_results/v3_task1ptdata_retr}"
HF=$YQQ/envs/n0vtla/bin/hf
PY=$YQQ/envs/n0vtla/bin/python

WHOAMI=$($HF auth whoami)
echo "hf auth whoami: $WHOAMI"
[[ "$WHOAMI" == *"qqyang"* ]] || { echo "ERROR: expected qqyang" >&2; exit 1; }

LABELS_IN="${LABELS:-base 20pct 40pct 60pct 80pct 100pct}"
OK=""
for l in $LABELS_IN; do
  if [[ -f "$RES/${l}_val.pt" && -f "$RES/${l}_train.pt" ]]; then OK="$OK $l"; else echo "SKIP $l: incomplete" >&2; fi
done
echo "aggregating:$OK"

rm -rf "$STAGE"; mkdir -p "$STAGE/embeddings" "$STAGE/logs" "$STAGE/per_checkpoint"
$PY scripts/aggregate_stage1_scaling_posttrain_data_eval.py --inputs "$RES" --labels $OK \
  --output-json "$STAGE/summary.json" --output-csv "$STAGE/summary.csv"
for l in $OK; do
  cp "$RES/${l}_val.pt" "$RES/${l}_train.pt" "$STAGE/embeddings/"
  cp "$RES/${l}_val.json" "$RES/${l}_train.json" "$STAGE/per_checkpoint/"
done
for jid in $EVAL_JOBIDS; do
  [[ -f "$YQQ/stage1_ptdata_eval_${jid}.log" ]] && cp "$YQQ/stage1_ptdata_eval_${jid}.log" "$STAGE/logs/eval_${jid}.log"
done

$HF upload qqyang/zihiao_real_test "$STAGE" "$HF_DEST" --repo-type dataset --token "$HF_TOKEN"
echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/${HF_DEST}"
rm -rf "$STAGE"
echo VISION_UPLOAD_PTDATA_DONE
