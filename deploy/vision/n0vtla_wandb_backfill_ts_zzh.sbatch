#!/bin/bash
#SBATCH --job-name=ts_wandb_bf
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/ts_wandb_bf_%j.log
#
# Incremental wandb backfill of the two TouchScale mid-trains (they run with wandb_enabled=False) from their
# segment logs. Re-submit any time; each run (<exp>-bf) only gets the steps it does not have yet.
# Submit from a checkout that has scripts/wandb_backfill_stage1_logs.py (cwd = that checkout).
set -uo pipefail
YQQ=/scratch/project/prj-02-phai-lab/yqq
source $YQQ/env.sh
PY=$YQQ/envs/n0vtla/bin/python
cd "$SLURM_SUBMIT_DIR"
MIN_JOB="${MIN_JOB:-991346}"   # first real segment; older logs are smokes
for EXP in ts_all_20k ts_old100_shuf_14k; do
  $PY scripts/wandb_backfill_stage1_logs.py --exp-name "$EXP" --min-job "$MIN_JOB" --logs $YQQ/logs/midtrain_ts_*.log
done
echo BACKFILL_DONE
