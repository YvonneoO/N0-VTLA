#!/bin/bash
#SBATCH --job-name=n0vtla-ts-upl
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/ts_upload_%j.log
# zzh (prj-02-phai-lab) twin of n0vtla_upload_midtrain_ckpts_siyuan.sbatch for the TouchScale mid-trains: uploads
# checkpoint steps (model.safetensors + optimizer.pt + metadata.pt, enough to --resume) to HF, verified by size,
# plus the files needed to continue the run elsewhere: the per-task-scale normalization table, the episode manifest
# and the split's sample index. Idempotent (steps already on HF with the same sizes are skipped).
# Env: EXP_NAME (required); STEPS (space-separated step folders, default: all complete ones);
#      SPLIT_KEY (default old100); REPO (default qqyang/zihiao_real_test); PREFIX (default n0vtla_touchscale_midtrain/<EXP_NAME>).
# Lines start with "UPL|".
set -euo pipefail
: "${EXP_NAME:?}"
YQQ=/scratch/project/prj-02-phai-lab/yqq
set +u; source $YQQ/env.sh; set -u
IN=$YQQ/data/touchscale_midtrain
export CKDIR="$YQQ/N0-VTLA_touchscale/checkpoints/vtla_stage1_predictor_pretrain/$EXP_NAME"
export REPO="${REPO:-qqyang/zihiao_real_test}" PREFIX="${PREFIX:-n0vtla_touchscale_midtrain/$EXP_NAME}"
export STEPS="${STEPS:-}"
export EXTRA="$IN/normalization_pad30_per_task_scale_touchscale_tacwam.json:normalization_pad30_per_task_scale_touchscale_tacwam.json $IN/manifest.json:data_manifest.json $IN/index_${SPLIT_KEY:-old100}.json:index_${SPLIT_KEY:-old100}.json"
$YQQ/envs/n0vtla/bin/python - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
assert api.whoami()["name"] == "qqyang", "HF token is not qqyang"
ck, repo, prefix = Path(os.environ["CKDIR"]), os.environ["REPO"], os.environ["PREFIX"]
NEED = ("model.safetensors", "optimizer.pt", "metadata.pt")
only = set(os.environ["STEPS"].split())

def remote_sizes(paths):
    return {i.path: i.size for i in api.get_paths_info(repo, paths, repo_type="dataset") if getattr(i, "size", None) is not None}

steps = sorted((p for p in ck.iterdir() if p.is_dir() and p.name.isdigit() and (not only or p.name in only)),
               key=lambda p: int(p.name))
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
    print(f"UPL|uploaded step {s.name} ({sum(files.values()) / 2**30:.2f} GiB): {sorted(files)}, verified", flush=True)
for item in os.environ["EXTRA"].split():
    local, name = item.split(":")
    if Path(local).exists():
        api.upload_file(path_or_fileobj=local, path_in_repo=f"{prefix}/{name}", repo_id=repo, repo_type="dataset",
                        commit_message=f"{os.path.basename(prefix)}: {name}")
        print(f"UPL|uploaded {name} ({Path(local).stat().st_size / 2**20:.1f} MiB)", flush=True)
print(f"UPL|{ck.name}: {len(steps)} step folders selected; uploaded {uploaded}; already on HF {done}; incomplete (skipped) {skipped}")
print(f"UPL|HF {repo}:{prefix}/  UPLOAD_DONE", flush=True)
PYEOF
