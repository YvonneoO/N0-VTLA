#!/bin/bash
#SBATCH --job-name=n0vtla-itw-inputs
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/itw_inputs_%j.log
# Mid-train inputs for the raw ITW tree synced from S3 (n0vtla_itw_s3_download_siyuan.sbatch):
#  1. dedicated clone yqq/N0-VTLA_itw (created on first run, fast-forwarded afterwards; ITW mid-train runs from it);
#  2. manifest.json + index_old100.json -> yqq/data/itw_midtrain (scripts/build_itw_raw_midtrain_inputs.py).
# Summary lines start with "ITB|". Env: ROOT (default yqq/data/raw), OUT (default yqq/data/itw_midtrain).
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
CLONE=$YQQ/N0-VTLA_itw
ROOT="${ROOT:-$YQQ/data/raw}"; OUT="${OUT:-$YQQ/data/itw_midtrain}"
PY=$YQQ/envs/n0vtla/bin/python
if [[ ! -d $CLONE/.git ]]; then git clone -q https://github.com/YvonneoO/N0-VTLA.git $CLONE; fi
cd $CLONE && git pull -q --ff-only && git log -1 --format='ITB|clone at %h %s'
export PYTHONPATH=$CLONE:$CLONE/scripts PYTHONDONTWRITEBYTECODE=1
$PY -c "import n0vtla; assert '$CLONE' in n0vtla.__file__, n0vtla.__file__; print('ITB|env ok, n0vtla from', n0vtla.__file__)"
$PY scripts/build_itw_raw_midtrain_inputs.py $ROOT $OUT --norm $YQQ/data/normalization_pad30_per_task_scale.json --workers 16
