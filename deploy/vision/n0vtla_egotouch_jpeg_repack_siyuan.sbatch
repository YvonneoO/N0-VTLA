#!/bin/bash
#SBATCH --job-name=n0vtla-egt-jpeg
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/egotouch_jpeg_%j.log
# Per-frame-JPEG repack of the EgoTouch 3-view@224 build for mid-train (scripts/repack_egotouch_jpeg.py):
#   1. size estimate on 60 sampled sequences; stop if the projection exceeds MAX_GIB (default 300);
#   2. full repack -> yqq/data/EgoTouch_jpeg224 (resumable; the source e2c_dl build is untouched);
#   3. manifest + index caches for the repack -> yqq/data/egotouch_midtrain_jpeg, then the CPU loader smoke on it.
# Summary lines start with "JPK|" / "EGB|" / "EGS|".
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
CLONE=$YQQ/N0-VTLA_egotouch
SRC=$YQQ/e2c_dl/EgoTouch_hdf5_3view_224
DST=$YQQ/data/EgoTouch_jpeg224
IN=$YQQ/data/egotouch_midtrain_jpeg
PY=$YQQ/envs/n0vtla/bin/python
cd $CLONE && git pull -q --ff-only && git log -1 --format='JPK|clone at %h %s'
export PYTHONPATH=$CLONE:$CLONE/scripts PYTHONDONTWRITEBYTECODE=1
$PY scripts/repack_egotouch_jpeg.py $SRC $DST --estimate 60 --workers 32 | tee /dev/stderr | grep -o 'GiB = [0-9.]* GiB projected' > $YQQ/logs/.egt_jpeg_proj_$SLURM_JOB_ID || true
PROJ=$(grep -o '[0-9.]* GiB projected' $YQQ/logs/.egt_jpeg_proj_$SLURM_JOB_ID | cut -d' ' -f1); rm -f $YQQ/logs/.egt_jpeg_proj_$SLURM_JOB_ID
echo "JPK|projected ${PROJ:-?} GiB (limit ${MAX_GIB:-300})"
if [[ -z "$PROJ" ]] || (( ${PROJ%.*} > ${MAX_GIB:-300} )); then echo "JPK|STOP: projection missing or above limit"; exit 1; fi
$PY scripts/repack_egotouch_jpeg.py $SRC $DST --workers 32
$PY scripts/build_egotouch_midtrain_inputs.py $DST $IN --split-file split_full.json --workers 16
$PY scripts/smoke_test_egotouch_loader.py $IN/manifest.json train_val $IN/index_train_val.json $IN/loader_smoke.png \
  --sequences 40 --samples 120
du -sh $DST | sed 's/^/JPK|du /'
