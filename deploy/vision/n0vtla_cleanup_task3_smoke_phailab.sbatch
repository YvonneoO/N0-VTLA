#!/bin/bash
#SBATCH --job-name=cleanup_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --account=prj-02-phai-lab
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/%x_%j.log
#
# Remove the checkpoints of the Whiteboard-erasing post-train SMOKE runs (20 steps, no longer needed; each step directory is on the order of ten GB) and the
# GT smoke spec. Only these exact names, directly under the checkpoint root / data root; nothing else, never a symlink. Prints sizes before removal.
# SUBMIT IT YOURSELF:  sbatch deploy/vision/n0vtla_cleanup_task3_smoke_phailab.sbatch
set -uo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
CK=$YQQ/N0-VTLA/checkpoints/vtla_tactile_posttrain
for d in "$CK/task3_zerotac0_smoke" "$CK/task3_zerotac0_smoke2" "$CK/task3_predtacq_smoke" "$CK/task3_predtacq_smoke2" "$YQQ/data/robot_ft_dense_smoke"; do
  case "$d" in
    "$CK"/task3_zerotac0_smoke|"$CK"/task3_zerotac0_smoke2|"$CK"/task3_predtacq_smoke|"$CK"/task3_predtacq_smoke2|"$YQQ/data/robot_ft_dense_smoke") ;;
    *) echo "ABORT unexpected path $d"; exit 3;;
  esac
  if [ ! -d "$d" ] || [ -L "$d" ]; then echo "skip $d (absent or a symlink)"; continue; fi
  echo "removing $d ($(du -sh "$d" | cut -f1))"
  rm -rf "$d" && echo "removed"
done
echo CLEANUP_SMOKE_DONE
