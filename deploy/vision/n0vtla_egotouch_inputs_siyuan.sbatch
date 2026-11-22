#!/bin/bash
#SBATCH --job-name=n0vtla-egt-inputs
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/egotouch_inputs_%j.log
# EgoTouch mid-train inputs on the siyuan account, then the CPU loader smoke:
#  1. dedicated clone yqq/N0-VTLA_egotouch (created on first run, fast-forwarded afterwards; training runs from it,
#     so the live checkout yqq/N0-VTLA is never pulled under a running job);
#  2. manifest.json + index_{train_val,test}.json -> yqq/data/egotouch_midtrain (scripts/build_egotouch_midtrain_inputs.py);
#  3. scripts/smoke_test_egotouch_loader.py on 40 train_val sequences -> contact sheet loader_smoke.png.
# Summary lines start with "EGB|" / "EGS|".
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
CLONE=$YQQ/N0-VTLA_egotouch
ROOT=$YQQ/e2c_dl/EgoTouch_hdf5_3view_224
OUT=$YQQ/data/egotouch_midtrain
PY=$YQQ/envs/n0vtla/bin/python
if [[ ! -d $CLONE/.git ]]; then git clone -q https://github.com/YvonneoO/N0-VTLA.git $CLONE; fi
cd $CLONE && git pull -q --ff-only && git log -1 --format='EGB|clone at %h %s'
export PYTHONPATH=$CLONE:$CLONE/scripts PYTHONDONTWRITEBYTECODE=1
$PY -c "import h5py, cv2, n0vtla; assert '$CLONE' in n0vtla.__file__, n0vtla.__file__; print('EGB|env ok, n0vtla from', n0vtla.__file__)"
$PY scripts/build_egotouch_midtrain_inputs.py $ROOT $OUT --split-file split_full.json --workers 8
$PY scripts/smoke_test_egotouch_loader.py $OUT/manifest.json train_val $OUT/index_train_val.json $OUT/loader_smoke.png \
  --sequences 40 --samples 120
