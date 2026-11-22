#!/bin/bash
#SBATCH --job-name=n0vtla-dl-sv2-0pct
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/dl_sv2_0pct_%j.log
#
# smoke_test_v2 "0%" reference for the held-out loss eval: n0vtla_wetlab_posttrain/checkpoint_20000 =
# official n0-vtla-base post-trained 20k steps on the OLD smoke_test data (asset wetlab_v2_train).
# Its own norm stats are copied to assets/vtla_tactile_posttrain/wetlab_v2_train so the eval can
# normalize with what the model was trained with. No optimizer.pt.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
DEST=$YQQ/data/n0vtla/eval_ckpts/sv2_0pct_old
$YQQ/envs/n0vtla/bin/hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0vtla_wetlab_posttrain/checkpoint_20000/*" --exclude "*optimizer.pt" \
  --local-dir $DEST --token "$HF_TOKEN"
SRC=$DEST/n0vtla_wetlab_posttrain/checkpoint_20000
ASSET=$YQQ/N0-VTLA/assets/vtla_tactile_posttrain/wetlab_v2_train
mkdir -p $ASSET
cp $SRC/assets/wetlab_v2_train/norm_stats.json $ASSET/norm_stats.json
ls -la $SRC $ASSET
echo DL_SV2_0PCT_DONE
