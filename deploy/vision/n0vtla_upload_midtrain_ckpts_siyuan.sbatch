#!/bin/bash
#SBATCH --job-name=n0vtla-midtrain-upl
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/midtrain_upload_%j.log
# Upload every checkpoint step of a mid-train run that is not yet on HF (model.safetensors + optimizer.pt + metadata.pt,
# so a run can be resumed from HF), plus wandb_id.txt and the data manifest. Idempotent: steps already on HF with the
# same file sizes are skipped; a step folder missing any of the three files (a save cut off by the walltime) is skipped.
# Meant to run after each training segment: sbatch --dependency=afterany:<segment> --export=ALL,EXP_NAME=egt_20k ...
# Env: EXP_NAME (required), CLONE (default yqq/N0-VTLA_egotouch), REPO (default qqyang/zihiao_real_test),
#      PREFIX (default n0vtla_egotouch_midtrain/<EXP_NAME>), MANIFEST (default yqq/data/egotouch_midtrain_jpeg/manifest.json).
#      ONLY_STEPS (space-separated step numbers; default all), EXTRAS (default 1; 0 = do not upload wandb_id.txt / manifest,
#      e.g. when adding a step to a folder whose run was started on another account).
# Lines start with "UPL|".
set -euo pipefail
: "${EXP_NAME:?}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
set +u; source $YQQ/env.sh; set -u
export CKDIR="${CLONE:-$YQQ/N0-VTLA_egotouch}/checkpoints/vtla_stage1_predictor_pretrain/$EXP_NAME"
export REPO="${REPO:-qqyang/zihiao_real_test}" PREFIX="${PREFIX:-n0vtla_egotouch_midtrain/$EXP_NAME}"
export MANIFEST="${MANIFEST:-$YQQ/data/egotouch_midtrain_jpeg/manifest.json}"
export ONLY_STEPS="${ONLY_STEPS:-}" EXTRAS="${EXTRAS:-1}"
$YQQ/envs/n0vtla/bin/python - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
assert api.whoami()["name"] == "qqyang", "HF token is not qqyang"
ck, repo, prefix = Path(os.environ["CKDIR"]), os.environ["REPO"], os.environ["PREFIX"]
NEED = ("model.safetensors", "optimizer.pt", "metadata.pt")

def remote_sizes(paths):
    return {i.path: i.size for i in api.get_paths_info(repo, paths, repo_type="dataset") if getattr(i, "size", None) is not None}

steps = sorted((p for p in ck.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
only = set(os.environ["ONLY_STEPS"].split())
if only:
    steps = [p for p in steps if p.name in only]
    print(f"UPL|ONLY_STEPS {sorted(only, key=int)} -> {[p.name for p in steps]}", flush=True)
done, skipped, uploaded = [], [], []
for s in steps:
    files = {f.name: f.stat().st_size for f in s.iterdir() if f.is_file()}
    if not all(n in files for n in NEED):
        skipped.append(s.name); continue
    want = {f"{prefix}/{s.name}/{n}": sz for n, sz in files.items()}
    if remote_sizes(list(want)) == want:
        done.append(s.name); continue
    api.upload_folder(folder_path=str(s), path_in_repo=f"{prefix}/{s.name}", repo_id=repo, repo_type="dataset",
                      commit_message=f"{os.path.basename(prefix)} step {s.name} (weights + optimizer state)")
    got = remote_sizes(list(want))
    assert got == want, f"step {s.name}: HF sizes {got} != local {want}"
    uploaded.append(s.name)
    print(f"UPL|uploaded step {s.name} ({sum(files.values()) / 2**30:.2f} GiB), verified", flush=True)
extra = [(ck / "wandb_id.txt", f"{prefix}/wandb_id.txt"), (Path(os.environ["MANIFEST"]), f"{prefix}/data_manifest.json")]
if os.environ["EXTRAS"] == "0":
    extra = []
for local, remote in extra:
    if local.exists():
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type="dataset",
                        commit_message=f"{os.path.basename(prefix)}: {local.name}")
print(f"UPL|{ck.name}: {len(steps)} step folders; uploaded {uploaded}; already on HF {len(done)}; incomplete (skipped) {skipped}")
print(f"UPL|HF {repo}:{prefix}/  UPLOAD_DONE", flush=True)
PYEOF
