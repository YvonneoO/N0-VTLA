#!/bin/bash
#SBATCH --job-name=n0vtla-dl-sv2-ref
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/dl_sv2_ref_%j.log
#
# smoke_test_v2 100% reference for the held-out loss eval: human_data_post_train_v2 @ step 10000
# (Stage-1 human 14000 -> post-train on smoke_test_v2, lab 6-GPU run, asset wetlab_smoketestv2_train).
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
$YQQ/envs/n0vtla/bin/hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_10000/*" \
  --local-dir $YQQ/data/n0vtla/eval_ckpts/sv2_100pct --token "$HF_TOKEN"
ls -la $YQQ/data/n0vtla/eval_ckpts/sv2_100pct/n0vtla_wetlab_posttrain/human_data_post_train_v2/checkpoint_10000
echo DL_SV2_REF_DONE
