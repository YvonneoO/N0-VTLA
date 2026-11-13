#!/bin/bash
#SBATCH --job-name=n0vtla-fetch-merge
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/fetch_merge_s1_%j.log
#
# Fetch ONE mid-train step that another account uploaded (e.g. zzh's TouchScale ts_all_20k/20000, n0vtla_upload_midtrain_ckpts_zzh.sbatch)
# from qqyang/zihiao_real_test, then merge it onto n0-vtla-base (scripts/merge_stage1_into_base_checkpoint.py) -> the post-train init.
# Env: SRC_EXP (required), STEP (required), OUT (required), PREFIX (default n0vtla_touchscale_midtrain/<SRC_EXP>),
#      CLONE (checkout with the merge script, default yqq/N0-VTLA_usb).
#   sbatch --export=ALL,SRC_EXP=ts_all_20k,STEP=20000,OUT=$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_ts_all_20000 <this>
set -euo pipefail
: "${SRC_EXP:?}"; : "${STEP:?}"; : "${OUT:?}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
set +u; source $YQQ/env.sh; set -u
export PREFIX="${PREFIX:-n0vtla_touchscale_midtrain/$SRC_EXP}" STEP
export DEST=$YQQ/data/ts_resume/$SRC_EXP/hf
$YQQ/envs/n0vtla/bin/python - <<'PYEOF'
import os, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.whoami()["name"] == "qqyang", "HF token is not qqyang"
repo, prefix, step, dest = "qqyang/zihiao_real_test", os.environ["PREFIX"], os.environ["STEP"], os.environ["DEST"]
for attempt in range(6):   # HF 429 (token rate limit, e.g. while a big download runs on the same token): pause and retry
    try:
        snapshot_download(repo, repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=dest, max_workers=2,
                          allow_patterns=[f"{prefix}/{step}/*", f"{prefix}/*.json"])
        break
    except Exception as e:
        if attempt == 5:
            raise
        print(f"FM|attempt {attempt + 1} failed ({type(e).__name__}: {str(e)[:120]}), retry in 5 min", flush=True)
        time.sleep(300)
d = Path(dest) / prefix / step
remote = {i.path.split("/")[-1]: i.size for i in api.get_paths_info(repo, [f"{prefix}/{step}/{n}" for n in ("model.safetensors", "optimizer.pt", "metadata.pt")], repo_type="dataset")}
local = {n: (d / n).stat().st_size for n in remote if (d / n).exists()}
assert local == remote, f"size mismatch HF {remote} local {local}"
print(f"FM|fetched {repo}:{prefix}/{step} -> {d} {local}")
PYEOF
STAGE1=$DEST/$PREFIX/$STEP
cd "${CLONE:-$YQQ/N0-VTLA_usb}"
for f in model.safetensors metadata.pt; do [[ -s "$STAGE1/$f" ]] || { echo "FM|missing $STAGE1/$f"; exit 1; }; done
$YQQ/envs/n0vtla/bin/python scripts/merge_stage1_into_base_checkpoint.py \
  --base-checkpoint $YQQ/data/n0vtla/n0-vtla-base --stage1-checkpoint "$STAGE1" --output "$OUT"
ls -la "$OUT"
echo "FM|merged $STAGE1 -> $OUT"
echo FETCH_MERGE_DONE
