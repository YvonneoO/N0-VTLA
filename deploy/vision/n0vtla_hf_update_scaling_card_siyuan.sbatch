#!/bin/bash
#SBATCH --job-name=n0vtla-hf-scaling-card
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/hf_scaling_card_%j.log
#
# (1) Moves the first-pass eval files that sit directly under
#     n0vtla_pretrain_scaling_study/held_out_eval_results/ into its v1_batchnce_recon/ subdir
#     (copy commit, then delete commit; no-op if already moved).
# (2) Uploads docs/hf_cards/n0vtla_pretrain_scaling_study.md as n0vtla_pretrain_scaling_study/README.md
#     and inserts/replaces the same text as a marked section of the repo's root README.md (dataset
#     card), leaving the rest of the card untouched. Idempotent: re-run after new results to refresh.
# Needs the repo checkout (docs/hf_cards/...) in N0-VTLA_scaling_extract.
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
cd $YQQ/N0-VTLA_scaling_extract

$YQQ/envs/n0vtla/bin/python - <<'PY'
import os, re, shutil, tempfile
from huggingface_hub import (HfApi, CommitOperationAdd, CommitOperationCopy,
                             CommitOperationDelete, hf_hub_download)
from huggingface_hub.errors import EntryNotFoundError

REPO = "qqyang/zihiao_real_test"
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()["name"]
assert who == "qqyang", who

base = "n0vtla_pretrain_scaling_study/held_out_eval_results"
new = f"{base}/v1_batchnce_recon"
files = api.list_repo_files(REPO, repo_type="dataset")
old = [f for f in files if f.startswith(base + "/")
       and not re.match(rf"{re.escape(base)}/v\d+_", f)]
if old:
    api.create_commit(REPO, repo_type="dataset", commit_message="Move v1 held-out eval into v1_batchnce_recon/",
                      operations=[CommitOperationCopy(src_path_in_repo=f, path_in_repo=new + f[len(base):]) for f in old])
    api.create_commit(REPO, repo_type="dataset", commit_message="Remove old flat v1 held-out eval files",
                      operations=[CommitOperationDelete(path_in_repo=f) for f in old])
    print(f"moved {len(old)} files into {new}")
else:
    print("nothing to move")

section = open("docs/hf_cards/n0vtla_pretrain_scaling_study.md").read().strip()
BEGIN, END = "<!-- BEGIN n0vtla_pretrain_scaling_study -->", "<!-- END n0vtla_pretrain_scaling_study -->"
block = f"{BEGIN}\n{section}\n{END}"
try:
    root = open(hf_hub_download(REPO, "README.md", repo_type="dataset", token=os.environ["HF_TOKEN"])).read()
except EntryNotFoundError:
    root = ""
if BEGIN in root:
    root = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda m: block, root, flags=re.S)
else:
    root = (root.rstrip() + "\n\n" if root.strip() else "") + block + "\n"
d = tempfile.mkdtemp(dir="/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data")
open(f"{d}/root_README.md", "w").write(root)
open(f"{d}/study_README.md", "w").write(section + "\n")
api.create_commit(REPO, repo_type="dataset", commit_message="Update scaling-study dataset card",
                  operations=[CommitOperationAdd("README.md", f"{d}/root_README.md"),
                              CommitOperationAdd("n0vtla_pretrain_scaling_study/README.md", f"{d}/study_README.md")])
shutil.rmtree(d)
print("card updated: https://huggingface.co/datasets/" + REPO)
PY
echo VISION_HF_SCALING_CARD_DONE
