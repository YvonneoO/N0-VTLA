#!/bin/bash
#SBATCH --job-name=n0vtla-du-audit
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/du_audit_%j.log
#
# Disk usage audit for the N0-VTLA working tree + related caches -- checkpoints,
# smoke-test/job logs, and wandb run data specifically. CPU-only, sbatch (never du
# interactively on the shared login node -- see @TAMU/CLAUDE.md's documented pitfall).
#
# Submit with: sbatch scripts/vision_du_audit.sbatch
set -euo pipefail

N0VTLA_ROOT=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA
WANDB_ROOT=/scratch/project/prj-02-phai-lab/yqq/.cache/wandb

echo "=== N0-VTLA total ==="
du -sh "$N0VTLA_ROOT" 2>/dev/null || true

echo
echo "=== N0-VTLA top-level breakdown ==="
du -sh "$N0VTLA_ROOT"/* 2>/dev/null | sort -rh || true

echo
echo "=== checkpoints/ by experiment ==="
du -sh "$N0VTLA_ROOT"/checkpoints/*/* 2>/dev/null | sort -rh || true

echo
echo "=== checkpoints/ by individual saved step (top 30) ==="
du -sh "$N0VTLA_ROOT"/checkpoints/*/*/* 2>/dev/null | sort -rh | head -30 || true

echo
echo "=== *.log at N0-VTLA root: count + total size ==="
find "$N0VTLA_ROOT" -maxdepth 1 -name '*.log' | wc -l
du -ch "$N0VTLA_ROOT"/*.log 2>/dev/null | tail -1 || true

echo
echo "=== *.log top 20 largest ==="
du -sh "$N0VTLA_ROOT"/*.log 2>/dev/null | sort -rh | head -20 || true

echo
echo "=== wandb cache total ==="
du -sh "$WANDB_ROOT" 2>/dev/null || true

echo
echo "=== wandb run count + top 20 largest runs ==="
find "$WANDB_ROOT/wandb" -maxdepth 1 -type d -name 'run-*' 2>/dev/null | wc -l
du -sh "$WANDB_ROOT"/wandb/run-* 2>/dev/null | sort -rh | head -20 || true

echo DU_AUDIT_COMPLETE
