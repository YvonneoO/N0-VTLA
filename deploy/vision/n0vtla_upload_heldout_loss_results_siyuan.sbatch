#!/bin/bash
#SBATCH --job-name=n0vtla-up-ho-loss
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/upload_ho_loss_%j.log
#
# Uploads the held-out action-loss results (per-checkpoint JSON + summary.csv + the eval job log) to
# qqyang/zihiao_real_test:n0vtla_scaling_posttrain/offline_eval/heldout_action_loss/ . Idempotent.
#   sbatch --dependency=afterany:<eval job> --export=ALL,EVAL_JOBID=<eval job> deploy/vision/n0vtla_upload_heldout_loss_results_siyuan.sbatch
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
: "${EVAL_JOBID:?EVAL_JOBID is required}"
RES="${RES_DIR:-$YQQ/data/scaling_posttrain_heldout_loss_v2}"
DEST="${HF_DEST:-n0vtla_scaling_posttrain/offline_eval/heldout_action_loss}"
STAGE=$YQQ/data/_stage_ho_loss_upload
rm -rf $STAGE; mkdir -p $STAGE/logs
cp $RES/*.json $RES/summary.csv $STAGE/
cp $YQQ/logs/n0vtla_ho_loss_${EVAL_JOBID}.log $STAGE/logs/
[[ "$($YQQ/envs/n0vtla/bin/hf auth whoami)" == *qqyang* ]] || { echo "HF account is not qqyang" >&2; exit 1; }
$YQQ/envs/n0vtla/bin/hf upload qqyang/zihiao_real_test $STAGE $DEST --repo-type dataset --token "$HF_TOKEN"
rm -rf $STAGE
echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/$DEST"
echo UPLOAD_HO_LOSS_DONE
