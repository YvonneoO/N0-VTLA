#!/bin/bash
#SBATCH --job-name=n0vtla-task-name-compare-aug
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/task_name_compare_aug_%j.log
#
# Follow-up to job 534596: July-vs-August names had 0/56 vs 0/55 overlap AND a
# qualitative style shift (July = task/action descriptions, August sample = bare
# object/item labels like "AAA 电池", "A4纸") -- but itw08-10 alone matched the
# per-task-scale table 100% (job 534585), whose own keys read as task descriptions,
# not object labels. So the "object label" style isn't uniform across all of August
# either. Compare itw08-03 (showed object-label style in job 534596's mixed sample)
# against itw08-10 (task-description style, matches the audit table) directly.
#
# Submit with: sbatch scripts/vision_compare_task_names_within_august.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
python scripts/compare_task_names_across_dates.py "$RAW_ROOT" \
  --dates-a itw08-03 --dates-b itw08-10 --sample 300
