#!/usr/bin/env bash
# Run inside n0vtla Docker, from the repository root. Existing outputs are protected.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
ROOT=/DATA2/qianqian
DATA="$ROOT/n0vtla_itw_canonical_smoke/itw0803_report_v1"
RUN=itw0803_pressure_report_v1_2000
REPORT="$ROOT/n0vtla_reports/$RUN"
NORM="$ROOT/n0vtla_data_manifests/normalization_0803_train_v1.json"
SPLIT="$ROOT/n0vtla_data_manifests/tacwam_v10_split.json"
export VTLA_PRETRAINED_CHECKPOINT="$ROOT/N0-VTLA/checkpoints/n0-vtla-base"
export VTLA_DATASET_PATH="$DATA/train" VTLA_ASSET_ID=itw0803_report_train_v1
mkdir -p "$REPORT"
python scripts/itw_tactile_smoke_adapter.py "$ROOT/n0vtla_itw_raw/itw08-03" "$DATA/train" \
    --normalization "$NORM" --split-manifest "$SPLIT" --split train --max-episodes 50
python scripts/itw_tactile_smoke_adapter.py "$ROOT/n0vtla_itw_raw/itw08-03" "$DATA/validation" \
    --normalization "$NORM" --split-manifest "$SPLIT" --split validation --max-episodes 10
python scripts/eval_stage1_report.py evaluate --dataset "$DATA/validation" \
    --checkpoint "$VTLA_PRETRAINED_CHECKPOINT" --label before --output "$REPORT/before"
python scripts/train_stage1_predictor.py vtla_stage1_predictor_pretrain \
    --exp-name="$RUN" --num-train-steps=2000 2>&1 | tee "$REPORT/training.log"
python scripts/eval_stage1_report.py evaluate --dataset "$DATA/validation" \
    --checkpoint "$ROOT/N0-VTLA/checkpoints/vtla_stage1_predictor_pretrain/$RUN/2000" \
    --label after --output "$REPORT/after"
python scripts/eval_stage1_report.py compare --dataset "$DATA/validation" \
    --before "$REPORT/before" --after "$REPORT/after" --output "$REPORT/comparison"
