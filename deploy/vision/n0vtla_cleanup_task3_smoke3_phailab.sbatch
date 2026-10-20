#!/bin/bash
#SBATCH --job-name=cleanup_smoke3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --account=prj-02-phai-lab
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/%x_%j.log
#
# Remove the checkpoints of the predicted-tactile Whiteboard-erasing post-train SMOKE (job 796177, 20 steps, steps 19 and 20, ~45 GB). One exact directory under the
# checkpoint root, never a symlink. Prints the size before removal.
# SUBMIT IT YOURSELF:  sbatch deploy/vision/n0vtla_cleanup_task3_smoke3_phailab.sbatch
set -uo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
d=$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain/task3_predtacq_smoke3
case "$d" in "$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain/task3_predtacq_smoke3") ;; *) echo "ABORT unexpected path $d"; exit 3;; esac
if [ ! -d "$d" ] || [ -L "$d" ]; then echo "skip $d (absent or a symlink)"; echo CLEANUP_SMOKE3_DONE; exit 0; fi
echo "removing $d ($(du -sh "$d" | cut -f1))"
rm -rf "$d" && echo "removed"
echo CLEANUP_SMOKE3_DONE
