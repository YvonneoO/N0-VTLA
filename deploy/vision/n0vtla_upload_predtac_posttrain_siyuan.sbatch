#!/bin/bash
#SBATCH --job-name=n0vtla-up-predtac
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/upload_predtac_posttrain_%x_%j.log
#
# Uploads the PREDICTED-TACTILE post-train checkpoints to qqyang/zihiao_real_test:n0vtla_predtac_posttrain/<exp>/<step>/ (own folder, with a README that
# says so). optimizer.pt is excluded (resume state, ~14 GB per step), as in the other post-train uploads. Submit as a watcher on the training job:
#   sbatch --dependency=afterok:<train jobid> --job-name=task1 --export=ALL,TASK=task1 deploy/vision/n0vtla_upload_predtac_posttrain_siyuan.sbatch
set -euo pipefail
: "${TASK:?TASK=task1|sv2}"
SUFFIX="${SUFFIX:-predtacq}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
EXP="${TASK}_posttrain_ours_${SUFFIX}_vision8gpu"
D=$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain/$EXP
[[ "$($HF auth whoami)" == *qqyang* ]] || { echo "ERROR: HF account is not qqyang" >&2; exit 1; }
[[ -d "$D" ]] || { echo "ERROR: $D missing (training did not write checkpoints)" >&2; exit 2; }
ls "$D"
$HF upload qqyang/zihiao_real_test "$YQQ/N0-VTLA/docs/hf_cards/n0vtla_predtac_posttrain.md" "n0vtla_predtac_posttrain/README.md" --repo-type dataset --token "$HF_TOKEN"
$HF upload qqyang/zihiao_real_test "$D" "n0vtla_predtac_posttrain/$EXP" --repo-type dataset --token "$HF_TOKEN" --exclude "*optimizer.pt" --exclude "*/optimizer.pt"
echo "https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/n0vtla_predtac_posttrain/$EXP"
echo UPLOAD_PREDTAC_POSTTRAIN_DONE
