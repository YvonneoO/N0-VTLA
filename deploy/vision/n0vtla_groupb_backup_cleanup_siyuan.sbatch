#!/bin/bash
#SBATCH --job-name=n0vtla-groupb-hf
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/groupb_hf_%j.log
# Free the siyuan disk of the finished Task-4 post-train runs ("group B", 2026-11-06 survey):
#  1. upload optimizer.pt of the LAST step of each run (HF already holds every step's weights + norm stats, and
#     optimizer state of the early steps only) to qqyang/zihiao_real_test under the run's existing folder;
#  2. check EVERY local checkpoint file against HF (same path, same size); optimizer.pt of non-last steps is
#     allowed to be missing on HF (not needed for inference/eval);
#  3. only if everything matched and DELETE=1 (default): rm -rf the 4 runs, the 3 post92traj smoke runs (never
#     uploaded, smokes) and data/n0vtla/eval_ckpts (local copies of HF reference checkpoints).
# Lines start with "GRB|".
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
set +u; source $YQQ/env.sh; set -u
export YQQ DELETE="${DELETE:-1}"
$YQQ/envs/n0vtla/bin/python - <<'PYEOF'
import os, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi

YQQ = Path(os.environ["YQQ"]); REPO = "qqyang/zihiao_real_test"
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()["name"]
assert who == "qqyang", f"HF token is {who!r}, expected qqyang"
print("GRB|HF user", who, flush=True)
CK = YQQ / "N0-VTLA/checkpoints/vtla_tactile_posttrain"
RUNS = {CK / f"task4_posttrain_ours_{v}_post92traj40k_vision8gpu":
        f"n0vtla_task4_dual_pour_posttrain/task4_posttrain_ours_{v}_post92traj40k_vision8gpu"
        for v in ("predtacq", "realtac", "zerotac0")}
RUNS[CK / "task4_posttrain_ours_zerotac0_vision8gpu"] = "n0vtla_notactile_posttrain/task4_posttrain_ours_zerotac0_vision8gpu"
EXTRA = [CK / f"task4_{v}_post92traj_smoke" for v in ("predtacq", "realtac", "zerotac0")] + [YQQ / "data/n0vtla/eval_ckpts"]

def hf_sizes(paths):
    out = {}
    for i in range(0, len(paths), 50):
        for info in api.get_paths_info(REPO, paths[i:i + 50], repo_type="dataset"):
            if getattr(info, "size", None) is not None:
                out[info.path] = info.size
    return out

all_ok = True
for local, prefix in RUNS.items():
    steps = sorted((p for p in local.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    last = steps[-1]
    opt = last / "optimizer.pt"
    remote_opt = f"{prefix}/{last.name}/optimizer.pt"
    have = hf_sizes([remote_opt])
    if have.get(remote_opt) == opt.stat().st_size:
        print(f"GRB|{local.name}: {last.name}/optimizer.pt already on HF", flush=True)
    else:
        print(f"GRB|{local.name}: uploading {last.name}/optimizer.pt ({opt.stat().st_size / 2**30:.1f} GiB)", flush=True)
        api.upload_file(path_or_fileobj=str(opt), path_in_repo=remote_opt, repo_id=REPO, repo_type="dataset",
                        commit_message=f"{local.name}: optimizer state of the last step {last.name}")
    files = [f for f in local.rglob("*") if f.is_file()]
    want = {f"{prefix}/{f.relative_to(local)}": f.stat().st_size for f in files}
    remote = hf_sizes(list(want))
    missing, mismatch, allowed = [], [], 0
    for path, size in want.items():
        rel = path[len(prefix) + 1:]
        if path not in remote:
            if rel.endswith("optimizer.pt") and not rel.startswith(last.name + "/"):
                allowed += 1
            elif rel.split("/")[0].isdigit():
                missing.append(rel)
            # top-level bookkeeping files (wandb_id.txt, .ddp_fs_ready) are not checkpoint content
        elif remote[path] != size:
            mismatch.append(rel)
    ok = not missing and not mismatch
    all_ok &= ok
    print(f"GRB|{local.name}: {len(files)} local files, missing on HF {missing[:5]} ({len(missing)}), size mismatch "
          f"{mismatch[:5]} ({len(mismatch)}), early-step optimizer states not on HF (allowed) {allowed} -> "
          f"{'OK' if ok else 'NOT OK'}", flush=True)

if not all_ok:
    print("GRB|NOT deleting anything: some checkpoint content is not on HF"); sys.exit(3)
if os.environ.get("DELETE") != "1":
    print("GRB|verified; DELETE != 1, nothing deleted"); sys.exit(0)
for p in list(RUNS) + EXTRA:
    if p.exists():
        shutil.rmtree(p)
        print(f"GRB|deleted {p}", flush=True)
print("GRB|GROUPB_DONE", flush=True)
PYEOF
