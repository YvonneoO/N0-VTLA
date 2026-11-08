#!/bin/bash
#SBATCH --job-name=n0vtla_st_build
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_st_build_%j.log
#
# The ORIGINAL smoke_test release (cap_to_tray, SingleBicycle/tacwam-wetlab-tasks:smoke_test/, not smoke_test_v2)
# -> canonical single-arm train/val datasets on VISION, prepared the same way as task1/2/3:
#   1. download smoke_test/** (gated; qqyang token) into data/n0vtla/wetlab_raw/smoke_test
#   2. split: the release's own training_split.json if it has one (translate_smoketestv2_split.py), else the
#      repo's scripts/wetlab_split_v1_tacwam_match.json (train 42 / val 5 source episodes, 6 block-holdout left out)
#   3. shared tactile normalization fit on the train split (fit_wetlab_tactile_norm.py)
#   4. canonical_wetlab_smoketest_{train,val} (build_wetlab_canonical_dataset.py: each causal contiguous segment of a
#      trajectory = one episode, as for task1/2/3)
#   5. state/action norm stats (rotation-aware compute_canonical_norm.py --robot wetlab_right_arm --train-only) ->
#      $YQQ/N0-VTLA/assets/vtla_tactile_posttrain/wetlab_smoketest_train/norm_stats.json (training runs from that checkout)
# Scripts run from the submit dir (the clean N0-VTLA_egotouch clone). Restart-safe (skips finished steps).
#   cd $YQQ/N0-VTLA_egotouch && sbatch deploy/vision/n0vtla_smoketest_build_siyuan.sbatch
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
set +u; source $YQQ/env.sh; set -u
N0VTLA_ENV=$YQQ/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV" CONDA_DEFAULT_ENV=n0vtla PATH="$N0VTLA_ENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu HF_HUB_ENABLE_HF_TRANSFER=1
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD:$PWD/scripts
PY=$N0VTLA_ENV/bin/python
DATA_ROOT=$YQQ/data/n0vtla
RAW_DIR=$DATA_ROOT/wetlab_raw/smoke_test
SPLIT=$DATA_ROOT/wetlab_split_smoketest_v1.json
NORM_PATH=$DATA_ROOT/wetlab_tactile_norm_smoketest_v1.json
ASSET_ID=wetlab_smoketest_train
SCRATCH=$DATA_ROOT/_scratch_smoketest
mkdir -p "$SCRATCH"

echo "=== [1/5] download smoke_test raw ==="
WHO=$(hf auth whoami 2>&1 | head -n 3)
echo "hf auth whoami: $WHO"
[[ "$WHO" == *qqyang* ]] || { echo "ERROR: HF token is not qqyang"; exit 1; }
hf download SingleBicycle/tacwam-wetlab-tasks --repo-type dataset --include "smoke_test/**" --local-dir "$DATA_ROOT/wetlab_raw" > /dev/null
echo "episode dirs: $(ls "$RAW_DIR" | grep -cE '^[0-9a-f-]{36}$')   other entries: $(ls "$RAW_DIR" | grep -vE '^[0-9a-f-]{36}$' | tr '\n' ' ')"

echo "=== [2/5] split ==="
if [[ -f "$SPLIT" ]]; then echo "exists: $SPLIT"
elif [[ -f "$RAW_DIR/training_split.json" ]]; then
  echo "release has training_split.json -> translate"; $PY scripts/translate_smoketestv2_split.py "$RAW_DIR/training_split.json" "$SPLIT"
else
  echo "no training_split.json in the release -> scripts/wetlab_split_v1_tacwam_match.json"; cp scripts/wetlab_split_v1_tacwam_match.json "$SPLIT"
fi
$PY - "$SPLIT" "$RAW_DIR" <<'EOF'
import json, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); raw = Path(sys.argv[2])
for k in ("train", "val"):
    miss = [u for u in m[k] if not (raw / u).is_dir()]
    print(f"split {k}: {len(m[k])} uuids, {len(miss)} not downloaded {miss[:3]}")
print("holdout:", len(m["block_holdout_v1"]["holdout_episodes"]))
EOF

echo "=== [3/5] tactile normalization (train split) ==="
[[ -f "$NORM_PATH" ]] && echo "exists, skipping" || $PY scripts/fit_wetlab_tactile_norm.py "$RAW_DIR" "$SPLIT" "$NORM_PATH"

echo "=== [4/5] canonical datasets ==="
for which in train val; do
  OUT=$DATA_ROOT/canonical_wetlab_smoketest_$which
  if [[ -f "$OUT/meta/info.json" ]]; then echo "exists: $OUT"; continue; fi
  rm -rf "$OUT"
  $PY scripts/build_wetlab_canonical_dataset.py "$RAW_DIR" "$SPLIT" "$NORM_PATH" "$OUT" --which $which --workers 16 --scratch-dir "$SCRATCH"
done
$PY - $DATA_ROOT/canonical_wetlab_smoketest_train $DATA_ROOT/canonical_wetlab_smoketest_val <<'EOF'
import json, sys
for d in sys.argv[1:]:
    i = json.load(open(f"{d}/meta/info.json"))
    print(f"STB|{d.rsplit('/', 1)[-1]}: {i['total_episodes']} episodes, {i['total_frames']} frames")
EOF

echo "=== [5/5] state/action norm stats ==="
$PY scripts/compute_canonical_norm.py --repo-id $DATA_ROOT/canonical_wetlab_smoketest_train --robot wetlab_right_arm \
  --train-config-name vtla_tactile_posttrain --asset-id $ASSET_ID --train-only --repo-root $YQQ/N0-VTLA
ls -l $YQQ/N0-VTLA/assets/vtla_tactile_posttrain/$ASSET_ID/norm_stats.json
rm -rf "$SCRATCH"
echo SMOKETEST_CANONICAL_BUILD_COMPLETE
