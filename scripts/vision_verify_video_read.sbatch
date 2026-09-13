#!/bin/bash
#SBATCH --job-name=n0vtla-verify-video
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/verify_video_read_%j.log
#
# Checks scripts/verify_online_video_read.py against a real tujian_v5_recent episode.
# CPU-only: torchcodec's default decoder is software (CPU) decode, and this is purely a
# frame-indexing correctness check, not a throughput benchmark -- no GPU needed.
#
# Submit with: sbatch scripts/vision_verify_video_read.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

EPISODE=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent/itw07-23/0021b7c0-b812-4ce5-9f08-31573c06977f
python scripts/verify_online_video_read.py "$EPISODE" --positions 0,10,50,80

# Also settle which task_info.json key (tacWAM's own "name" vs our read_task's "steps"
# preference) real raw data actually uses -- needed to key the per-task tactile-scale
# normalization lookup correctly. See scripts/inspect_task_info_schema.py.
python scripts/inspect_task_info_schema.py "$EPISODE"
