#!/bin/bash
# Submits the whole Task3 pipeline with SLURM dependencies:
# download(563576, done) -> build -> norm stats -> {smoke ours, smoke baseline} -> {train ours, train baseline} -> watcher.
# Run from the N0-VTLA checkout on VISION: bash deploy/vision/submit_task3_chain.sh
set -euo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA
D=deploy/vision
BUILD=$(sbatch --parsable $D/n0vtla_task3_build_canonical.sbatch)
NORM=$(sbatch --parsable --dependency=afterok:$BUILD $D/n0vtla_task3_compute_norm_stats.sbatch)
SO=$(sbatch --parsable --dependency=afterok:$NORM $D/n0vtla_smoke_task3_ours.sbatch)
SB=$(sbatch --parsable --dependency=afterok:$NORM $D/n0vtla_smoke_task3_officialbase.sbatch)
TO=$(sbatch --parsable --dependency=afterok:$SO $D/n0vtla_train_task3_ours_1x8.sbatch)
TB=$(sbatch --parsable --dependency=afterok:$SB $D/n0vtla_train_task3_baseline_1x8.sbatch)
W=$(sbatch --parsable --dependency=after:$TO:$TB --export=ALL,OURS_JOBID=$TO,BASELINE_JOBID=$TB $D/n0vtla_watch_upload_task3_ckpts.sbatch)
echo "TASK3_CHAIN build=$BUILD norm=$NORM smoke_ours=$SO smoke_base=$SB train_ours=$TO train_base=$TB watcher=$W"
