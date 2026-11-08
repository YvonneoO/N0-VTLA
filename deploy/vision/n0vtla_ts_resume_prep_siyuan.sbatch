#!/bin/bash
#SBATCH --job-name=n0vtla-ts-resume-prep
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/ts_resume_prep_%j.log
# Continue a zzh TouchScale mid-train (e.g. B = ts_old100_shuf_14k) on siyuan's raw S3 copy of the ITW data:
#  1. download its checkpoint step + tacWAM-extended task-scale table + episode manifest from HF
#     (qqyang/zihiao_real_test:n0vtla_touchscale_midtrain/<SRC_EXP>/, uploaded by n0vtla_upload_midtrain_ckpts_zzh.sbatch);
#  2. place the step under N0-VTLA_itw/checkpoints/vtla_stage1_predictor_pretrain/<EXP>/<STEP> (and <SMOKE_EXP>/<STEP>),
#     so n0vtla_midtrain_itw_siyuan.sbatch resumes from it;
#  3. build manifest.json + index_<KEY>.json in OUT for the SAME episodes (uuids of the zzh list found on this copy).
# Env: SRC_EXP (default ts_old100_shuf_14k), STEP (default 8000), EXP (default <SRC_EXP>_siyuan),
#      SMOKE_EXP (default smoke_ts_resume), KEY (default old100), OUT (default yqq/data/ts_resume/<SRC_EXP>).
# Summary lines start with "PREP|" / "ITB|". Run from the idle yqq/N0-VTLA_egotouch checkout.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
set +u; source $YQQ/env.sh; set -u
PY=$YQQ/envs/n0vtla/bin/python
SRC_EXP="${SRC_EXP:-ts_old100_shuf_14k}"; STEP="${STEP:-8000}"; EXP="${EXP:-${SRC_EXP}_siyuan}"
SMOKE_EXP="${SMOKE_EXP:-smoke_ts_resume}"; KEY="${KEY:-old100}"
export OUT="${OUT:-$YQQ/data/ts_resume/$SRC_EXP}" PREFIX="n0vtla_touchscale_midtrain/$SRC_EXP" STEP
mkdir -p "$OUT"
$PY - <<'PYEOF'
import os
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.whoami()["name"] == "qqyang", "HF token is not qqyang"
p, s = os.environ["PREFIX"], os.environ["STEP"]
d = snapshot_download("qqyang/zihiao_real_test", repo_type="dataset", token=os.environ["HF_TOKEN"],
                      local_dir=os.path.join(os.environ["OUT"], "hf"), allow_patterns=[f"{p}/{s}/*", f"{p}/*.json"])
print("PREP|downloaded", sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs
                                 if not r.split(os.sep).count(".cache")), flush=True)
PYEOF
SRC="$OUT/hf/$PREFIX"
for f in model.safetensors optimizer.pt metadata.pt; do [[ -s "$SRC/$STEP/$f" ]] || { echo "PREP|missing $SRC/$STEP/$f"; exit 1; }; done
NORM="$SRC/normalization_pad30_per_task_scale_touchscale_tacwam.json"; LIST="$SRC/data_manifest.json"
[[ -s "$NORM" && -s "$LIST" ]] || { echo "PREP|missing norm or manifest under $SRC"; exit 1; }
for E in "$EXP" "$SMOKE_EXP"; do
  CK=$YQQ/N0-VTLA_itw/checkpoints/vtla_stage1_predictor_pretrain/$E
  if [[ -d "$CK" ]]; then echo "PREP|$CK already exists, left as is: $(ls "$CK" | tr '\n' ' ')"
  else mkdir -p "$CK/$STEP" && cp "$SRC/$STEP"/* "$CK/$STEP"/ && echo "PREP|placed step $STEP at $CK"; fi
done
CLONE=$YQQ/N0-VTLA_egotouch
cd $CLONE && git pull -q --ff-only && git log -1 --format='PREP|builder from %h %s'
export PYTHONPATH=$CLONE:$CLONE/scripts PYTHONDONTWRITEBYTECODE=1
$PY scripts/build_itw_raw_midtrain_inputs.py $YQQ/data/raw "$OUT" --norm "$NORM" --key "$KEY" \
    --episode-list "$LIST" --list-key "$KEY" --workers 16
echo "PREP|NORM=$NORM IN=$OUT EXP=$EXP SMOKE_EXP=$SMOKE_EXP  TS_RESUME_PREP_DONE"
