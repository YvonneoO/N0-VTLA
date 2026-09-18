#!/bin/bash
#SBATCH --job-name=n0vtla-dl-100pct-ckpt
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/dl_100pct_stage1_ckpt_%j.log
#
# Downloads the already-completed 100%-data Stage-1 checkpoint (trainable-only,
# step 14000, trained on prj-02-phai-lab's tujian_v5_recent corpus via
# vision_train_stage1_online_full.sbatch -- the "online_full_v3" wandb run) from HF
# into this account's own checkpoints/ tree, laid out the same way the scaling-study
# checkpoints are, so vision_stage1_scaling_held_out_eval_siyuan.sbatch can score it
# unmodified with EXP_NAME=stage1_online_100pct_phailab.
#
# CAVEAT (read before interpreting its held-out score next to the 20/40/60/80pct
# points): this checkpoint trained on a DIFFERENT raw corpus than this account's own
# yqq/data/raw (phai-lab's tujian_v5_recent vs. this account's own pull -- same ~39
# dates but a handful differ, see build_itw_scaling_splits.py's docstring). The
# held_out split used for every other point is a subset of THIS account's own corpus,
# so this checkpoint was almost certainly never trained on those exact episodes, but
# it wasn't drawn from the SAME shuffle/pool the other 4 points were, either -- a
# distribution-comparability caveat, not a leakage risk.
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract

DEST=checkpoints/vtla_stage1_predictor_pretrain/stage1_online_100pct_phailab/14000
mkdir -p "$DEST"

/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla/bin/hf download qqyang/zihiao_real_test \
  --repo-type dataset \
  --include "n0-vtla_ts_pretrain/14000/*" \
  --local-dir /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_dl_100pct_stage1 \
  --token "$HF_TOKEN"

mv /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_dl_100pct_stage1/n0-vtla_ts_pretrain/14000/* "$DEST"/
rm -rf /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/_dl_100pct_stage1

echo "Downloaded into $DEST:"
ls -la "$DEST"
echo VISION_DL_100PCT_STAGE1_CKPT_DONE
