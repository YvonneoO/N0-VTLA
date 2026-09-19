#!/bin/bash
#SBATCH --job-name=n0vtla-dl-smoketestv2
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/dl_smoketestv2_%j.log
#
# Pulls the lab-built canonical smoke_test_v2 datasets (59 train / 7 dev episodes, from the
# 60-episode HF release SingleBicycle/tacwam-wetlab-tasks:smoke_test_v2) plus its norm stats
# (asset id wetlab_smoketestv2_train) from qqyang/zihiao_real_test, and drops the norm stats
# where train.sh's preflight looks for them: N0-VTLA/assets/vtla_tactile_posttrain/<asset id>/.
# Same path task1's canonical data took (n0vtla_data_download.sbatch).
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
DATA_ROOT=$YQQ/data/n0vtla

[[ "$($HF auth whoami)" == *qqyang* ]] || { echo "ERROR: HF account is not qqyang" >&2; exit 1; }

$HF download qqyang/zihiao_real_test --repo-type dataset \
  --include "canonical_wetlab_smoketestv2_train/**" "canonical_wetlab_smoketestv2_dev/**" "wetlab_smoketestv2_train_norm_stats.json" \
  --local-dir "$DATA_ROOT" --token "$HF_TOKEN"

ASSET_DIR=$YQQ/N0-VTLA/assets/vtla_tactile_posttrain/wetlab_smoketestv2_train
mkdir -p "$ASSET_DIR"
cp "$DATA_ROOT/wetlab_smoketestv2_train_norm_stats.json" "$ASSET_DIR/norm_stats.json"

ls "$DATA_ROOT/canonical_wetlab_smoketestv2_train" "$DATA_ROOT/canonical_wetlab_smoketestv2_dev"
ls -la "$ASSET_DIR"
echo DL_SMOKETESTV2_DONE
