#!/bin/bash
#SBATCH --job-name=n0vtla-cleanup
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/cleanup_%j.log
#
# Deletes: (1) the smoke/diagnostic-test checkpoint dirs identified by the disk audit
# (job 537329) -- all already reported, not part of any active resume chain; (2) every
# *.log at the N0-VTLA repo root (except this job's own output log, which is still open
# for writing while this script runs -- unlinking it mid-job would silently lose this
# audit trail once the job exits and the fd closes); (3) the local wandb run cache
# (/scratch/project/prj-02-phai-lab/yqq/.cache/wandb) -- this is LOCAL staging/offline-
# sync cache only, not the actual synced data (that lives on wandb.ai), and it is
# shared across whatever other projects use this account's yqq/.cache, not N0-VTLA-only.
#
# Explicitly enumerated paths only -- no broad wildcard rm under checkpoints/, so the
# real training chains' own directories (online_full_v2, online_2gpu_v2,
# online_4gpu_2h_v1, online_4gpu_4h_v1) and checkpoints/n0-vtla-base are never touched
# even if they exist by the time this runs.
#
# Submit with: sbatch scripts/vision_cleanup_smoke_artifacts.sbatch
set -euo pipefail

N0VTLA_ROOT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA
WANDB_ROOT=/scratch/project/prj-02-phai-lab/yqq/.cache/wandb
CKPT_ROOT="$N0VTLA_ROOT/checkpoints/vtla_stage1_predictor_pretrain"

echo "=== BEFORE ==="
# `showquota` is a login-shell alias/function, not on PATH in a plain sbatch shell --
# first attempt (job 537399) aborted here under set -e before deleting anything.
# du -sh on the actual paths is more directly relevant to this job anyway.
du -sh "$N0VTLA_ROOT" 2>/dev/null || true

echo
echo "=== deleting smoke-test checkpoint dirs ==="
for name in online_realbs_535038 online_stability_534668 online_smoke_534645 online_fullcorpus_smoke online_full_v1; do
  dir="$CKPT_ROOT/$name"
  if [ -d "$dir" ]; then
    du -sh "$dir"
    rm -rf "$dir"
    echo "removed: $dir"
  else
    echo "already absent: $dir"
  fi
done

echo
echo "=== deleting *.log at N0-VTLA root (except this job's own output log) ==="
find "$N0VTLA_ROOT" -maxdepth 1 -name '*.log' -not -name "cleanup_${SLURM_JOB_ID}.log" -print -delete

echo
echo "=== deleting wandb local run cache ==="
if [ -d "$WANDB_ROOT" ]; then
  du -sh "$WANDB_ROOT"
  rm -rf "$WANDB_ROOT"
  echo "removed: $WANDB_ROOT"
else
  echo "already absent: $WANDB_ROOT"
fi

echo
echo "=== AFTER ==="
du -sh "$N0VTLA_ROOT" 2>/dev/null || true

echo CLEANUP_COMPLETE
