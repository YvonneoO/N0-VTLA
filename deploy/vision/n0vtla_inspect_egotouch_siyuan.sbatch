#!/bin/bash
#SBATCH --job-name=n0vtla-egt-insp
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/egotouch_inspect_%j.log
# Read-only survey of the EgoTouch 3-view@224 HDF5 build for N0-VTLA mid-train
# (scripts/inspect_egotouch_for_midtrain.py). Summary lines start with "EGT|"; full JSON -> OUT.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
ROOT=${ROOT:-$YQQ/e2c_dl/EgoTouch_hdf5_3view_224}
OUT=${OUT:-$YQQ/data/egotouch_midtrain_survey/survey.json}
cd $YQQ/N0-VTLA
export PYTHONDONTWRITEBYTECODE=1
$YQQ/envs/touchanything/bin/python scripts/inspect_egotouch_for_midtrain.py "$ROOT" "$OUT" --workers 8
du -sh "$ROOT" | sed 's/^/EGT|D /'
