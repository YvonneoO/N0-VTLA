#!/bin/bash
#SBATCH --job-name=n0vtla_usb_build
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_usb_build_%j.log
#
# Dual-arm USB plug (T10) post-train dataset, REAL glove tactile, all CPU. Same recipe as the task4 post92traj realtac dataset
# (Ego2Contact scripts/tools/dual42_build_siyuan.sbatch ARM=gt SET=b1b2 WHOLE=1):
#  [1/6] post-train split: the "train" lists of both batches (57 + 43 = 100 uuids)
#  [2/6] the 100 original episode dirs symlinked into one flat release dir
#  [3/6] pad30 tactile normalization fitted on these 100 (fit_wetlab_dual_tactile_norm.py; the pour table does not apply:
#        other task, and the pour gloves were stored crossed, USB is straight -- n0_dual_adapter.TASK_PROFILES)
#  [4/6] canonical dataset, --whole-trajectory (one episode per trajectory, interior gaps held, tac_0 = trajectory start)
#  [5/6] state/action norm stats, both arms relative (compute_canonical_norm.py --robot wetlab_dual_arm --train-only)
#  [6/6] read-back check through the real N0 input pipeline (verify_wetlab_dual_dataset.py)
# -> data/n0vtla/canonical_wetlab_usb_dual_post100traj_realtac, asset usb_plug_dual_post100traj (TASK=usb in the post-train launcher).
# Env: CLONE (N0-VTLA checkout with the USB-aware adapter, default yqq/N0-VTLA_usb), LIMIT (smoke: first N uuids, writes ..._smoke).
# Push-ramp frames of batch 2 (action/<side>/arm_perturb_ramp) stay in: the post-train loss does not read action_mask, as for task4.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export PATH="$YQQ/envs/n0vtla/bin:$PATH"
PY=$YQQ/envs/n0vtla/bin/python
DATA=$YQQ/data/n0vtla
RAW=$DATA/wetlab_raw
REL1=task_usb_plug_dual_20261108; REL2=task_usb_plug_dual_20261110
SPLIT=$DATA/training_split_usb_dual_post100.json
FLAT=$DATA/wetlab_raw_usb_dual_flat
NORM=$DATA/wetlab_tactile_norm_usb_dual.json
NAME=post100traj; ASSET=usb_plug_dual_post100traj; SUF=realtac
OUT=$DATA/canonical_wetlab_usb_dual_${NAME}_$SUF
if [ -n "${LIMIT:-}" ]; then OUT=${OUT}_smoke; ASSET=${ASSET}_smoke; fi
SCRATCH=$DATA/_scratch_usb_dual_${NAME}_$SUF${LIMIT:+_smoke}; mkdir -p $SCRATCH
cd "${CLONE:-$YQQ/N0-VTLA_usb}"
echo "=== clone $(pwd) @ $(git rev-parse --short HEAD)"
export PYTHONPATH=$PWD:$PWD/scripts

echo "=== [1/6] post-train split -> $SPLIT"
$PY - <<PYEOF
import json
from pathlib import Path
uuids = []
for rel in ("$REL1", "$REL2"):
    sp = json.load(open(Path("$RAW") / rel / "training_split.json"))
    tr = [x if isinstance(x, str) else x["uuid"] for x in sp["train"]]
    print(rel, len(tr), "train")
    uuids += tr
assert len(uuids) == len(set(uuids)) == 100, (len(uuids), len(set(uuids)))
out = dict(schema="wetlab_episode_window_split_v1", task="usb_plug_dual", train=sorted(uuids), validation=[], test=[],
           sources=["$REL1", "$REL2"], note="union of both batches' train lists; no held-out split (physical rollout is the evaluation)")
Path("$SPLIT").write_text(json.dumps(out, indent=1))
print("split:", len(uuids), "train uuids")
PYEOF

echo "=== [2/6] symlink the 100 episodes into $FLAT"
mkdir -p $FLAT
$PY - <<PYEOF
import json, os
from pathlib import Path
for u in json.load(open("$SPLIT"))["train"]:
    for rel in ("$REL1", "$REL2"):
        src = Path("$RAW") / rel / u
        if src.is_dir():
            dst = Path("$FLAT") / u
            if not dst.exists():
                os.symlink(src, dst)
            break
    else:
        raise SystemExit("episode not found: " + u)
print(len(os.listdir("$FLAT")), "episodes in", "$FLAT")
PYEOF

echo "=== [3/6] tactile normalization -> $NORM"
[ -f $NORM ] && echo "exists, skipping" || $PY scripts/fit_wetlab_dual_tactile_norm.py $FLAT $SPLIT $NORM

echo "=== [4/6] canonical dataset -> $OUT"
[ -d "$OUT" ] && echo "exists, skipping" || \
  $PY scripts/build_wetlab_dual_canonical_dataset.py $FLAT $SPLIT $NORM $OUT --workers 16 --scratch-dir $SCRATCH --whole-trajectory ${LIMIT:+--limit $LIMIT}

echo "=== [5/6] state/action norm stats ($ASSET) -> $YQQ/N0-VTLA/assets/vtla_tactile_posttrain/$ASSET"
# written into the training clone (yqq/N0-VTLA, --repo-root), where the post-train launcher reads assets from
[ -f "$YQQ/N0-VTLA/assets/vtla_tactile_posttrain/$ASSET/norm_stats.json" ] && echo "exists, skipping" || \
  $PY scripts/compute_canonical_norm.py --repo-id $OUT --robot wetlab_dual_arm --repo-root $YQQ/N0-VTLA \
    --train-config-name vtla_tactile_posttrain --asset-id $ASSET --train-only

echo "=== [6/6] verify"
C="$(pwd)"
( cd $YQQ/N0-VTLA && $PY $C/scripts/verify_wetlab_dual_dataset.py --dataset $OUT --asset-id $ASSET )   # assets are read cwd-relative (./assets)
$PY - <<PYEOF
import json
from pathlib import Path
m = Path("$OUT") / "meta"
eps = [json.loads(l) for l in (m / "episodes.jsonl").open()]
info = json.loads((m / "info.json").read_text())
print("episodes", len(eps), "frames", info.get("total_frames"), "task", (m / "tasks.jsonl").read_text().strip()[:200])
print("per-batch episodes:", {r: sum((Path("$RAW") / r / e["source_uuid"]).is_dir() for e in eps) for r in ("$REL1", "$REL2")})
PYEOF
echo USB_DUAL_BUILD_DONE
