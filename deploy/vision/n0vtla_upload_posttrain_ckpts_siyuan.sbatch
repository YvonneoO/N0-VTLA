#!/bin/bash
#SBATCH --job-name=n0vtla-pt-upl
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/pt_upload_%j.log
# Incremental upload of a post-train run (yqq/N0-VTLA/checkpoints/vtla_tactile_posttrain/<EXP>) to qqyang/zihiao_real_test:
# every complete step (model.safetensors + metadata.pt + assets/) WITHOUT optimizer.pt, plus optimizer.pt of the LATEST
# step only (resume state, ~14 GB), plus wandb_id.txt. Idempotent: files already on HF with the same size are skipped.
# Run after each training segment: sbatch --dependency=afterany:<segment> --export=ALL,EXP=<exp> <this>
# README: if the folder's README.md does not mention EXP, the text of NOTE_FILE (path relative to the submit dir) is
#   appended to it -- never overwritten.
# Env: EXP (required), PREFIX (default n0vtla_task4_dual_pour_posttrain), NOTE_FILE, EVERY + FINAL (upload only steps that
#   are multiples of EVERY or FINAL-1, e.g. EVERY=5000 FINAL=40000 with a pruned SAVE=1000 run; unset = every step).
#   optimizer.pt goes up with the latest uploaded step only if it still has one locally. Lines start with "PTU|".
set -euo pipefail
: "${EXP:?}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
set +u; source $YQQ/env.sh; set -u
export EVERY="${EVERY:-}" FINAL="${FINAL:-0}"
export CKDIR=$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain/$EXP PREFIX="${PREFIX:-n0vtla_task4_dual_pour_posttrain}"
export NOTE=""; [[ -n "${NOTE_FILE:-}" ]] && NOTE="$(cat "$SLURM_SUBMIT_DIR/$NOTE_FILE")"
$YQQ/envs/n0vtla/bin/python - <<'PYEOF'
import os, tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "qqyang/zihiao_real_test"
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.whoami()["name"] == "qqyang", "HF token is not qqyang"
ck, prefix, exp = Path(os.environ["CKDIR"]), os.environ["PREFIX"], os.environ["EXP"]
steps = sorted((p for p in ck.iterdir() if p.is_dir() and p.name.isdigit()
                and (p / "model.safetensors").is_file() and (p / "metadata.pt").is_file()), key=lambda p: int(p.name))
every, final = int(os.environ["EVERY"] or 0), int(os.environ["FINAL"])
if every:
    steps = [p for p in steps if int(p.name) % every == 0 or int(p.name) == final - 1]
if not steps:
    print(f"PTU|no complete step to upload yet in {ck}"); print("PT_UPLOAD_DONE"); raise SystemExit(0)
last = steps[-1].name

def remote_sizes(paths):
    out = {}
    for i in range(0, len(paths), 50):
        for info in api.get_paths_info(REPO, paths[i:i + 50], repo_type="dataset"):
            if getattr(info, "size", None) is not None:
                out[info.path] = info.size
    return out

for s in steps:
    files = [f for f in s.rglob("*") if f.is_file() and (f.name != "optimizer.pt" or s.name == last)]
    want = {f"{prefix}/{exp}/{f.relative_to(ck)}": f.stat().st_size for f in files}
    have = remote_sizes(list(want))
    todo = [f for f in files if have.get(f"{prefix}/{exp}/{f.relative_to(ck)}") != f.stat().st_size]
    if not todo:
        print(f"PTU|step {s.name}: already on HF", flush=True); continue
    ignore = None if s.name == last else ["optimizer.pt", "**/optimizer.pt"]
    api.upload_folder(folder_path=str(s), path_in_repo=f"{prefix}/{exp}/{s.name}", repo_id=REPO, repo_type="dataset",
                      ignore_patterns=ignore, commit_message=f"{exp} step {s.name}")
    got = remote_sizes(list(want))
    bad = [p for p, sz in want.items() if got.get(p) != sz]
    assert not bad, f"step {s.name}: not on HF with the local size: {bad[:4]}"
    print(f"PTU|step {s.name}: uploaded {len(todo)} files{' incl. optimizer.pt' if s.name == last else ''}, verified", flush=True)
w = ck / "wandb_id.txt"
if w.is_file():
    api.upload_file(path_or_fileobj=str(w), path_in_repo=f"{prefix}/{exp}/wandb_id.txt", repo_id=REPO, repo_type="dataset",
                    commit_message=f"{exp}: wandb_id.txt")
note = os.environ["NOTE"].strip()
if note:
    with tempfile.TemporaryDirectory(dir=str(ck)) as td:
        readme = Path(hf_hub_download(REPO, f"{prefix}/README.md", repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=td))
        text = readme.read_text()
        if exp not in text:
            readme.write_text(text.rstrip("\n") + "\n\n" + note + "\n")
            api.upload_file(path_or_fileobj=str(readme), path_in_repo=f"{prefix}/README.md", repo_id=REPO, repo_type="dataset",
                            commit_message=f"README: add {exp}")
            print("PTU|README: note appended", flush=True)
print(f"PTU|{exp}: {len(steps)} steps on HF (latest {last}) -> https://huggingface.co/datasets/{REPO}/tree/main/{prefix}/{exp}")
print("PT_UPLOAD_DONE")
PYEOF
