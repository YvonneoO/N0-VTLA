#!/bin/bash
#SBATCH --job-name=yqq-disk-survey
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/disk_survey_%j.log
# Size (bytes + file count) and last-modified date of everything in the siyuan yqq folder, two levels deep, for
# deciding what to archive to HF and delete. Read-only. Report -> yqq/logs/disk_survey_<job>.tsv
# (columns: GiB, files, newest mtime, path), largest first; the log holds the same lines prefixed "DSK|".
set -uo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
OUT=$YQQ/logs/disk_survey_$SLURM_JOB_ID.tsv
cd $YQQ
{
  for d in * .[!.]*; do
    [[ -e "$d" ]] || continue
    echo "$d"
    if [[ -d "$d" && ! -L "$d" ]]; then for s in "$d"/* "$d"/.[!.]*; do [[ -e "$s" ]] && echo "$s"; done; fi
  done
} | while read -r p; do
  b=$(du -sb "$p" 2>/dev/null | cut -f1); n=$(du -s --inodes "$p" 2>/dev/null | cut -f1)
  m=$(find "$p" -maxdepth 2 -printf '%TY-%Tm-%Td\n' 2>/dev/null | sort | tail -1)
  awk -v b="$b" -v n="$n" -v m="$m" -v p="$p" 'BEGIN{printf "%.1f\t%s\t%s\t%s\n", b/2^30, n, m, p}'
done | sort -t$'\t' -k1,1 -gr > $OUT
awk -F'\t' '$1 >= 1 || $2 >= 20000' $OUT | sed 's/^/DSK|/'
showquota | sed 's/^/DSK|/'
