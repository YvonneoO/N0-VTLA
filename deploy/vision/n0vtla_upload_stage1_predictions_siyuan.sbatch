#!/bin/bash
#SBATCH --job-name=n0vtla-up-s1preds
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/upload_s1preds_%j.log
#
# Uploads the per-sample prediction files (data/stage1_predictions/<task>_<split>.pt, ~2.6 GB) written by
# collect_stage1_multi_ckpt_predictions.py to
# qqyang/zihiao_real_test:n0vtla_pretrain_scaling_study/held_out_eval_results/v4_multitask_predictions/
# (smoke files excluded) so the metrics can be recomputed anywhere without re-running the model.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
DEST="${HF_DEST:-n0vtla_pretrain_scaling_study/held_out_eval_results/v4_multitask_predictions}"
[[ "$($HF auth whoami)" == *qqyang* ]] || { echo "HF account is not qqyang" >&2; exit 1; }
$HF upload qqyang/zihiao_real_test $YQQ/data/stage1_predictions "$DEST" --repo-type dataset --token "$HF_TOKEN" --exclude "*smoke*"
echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/$DEST"
echo UPLOAD_S1PREDS_DONE
