#!/bin/bash
#SBATCH --job-name=n0vtla-online-smoke
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/online_loader_smoke_%j.log
#
# Real (non-mocked) smoke test of ITWOnlineTactileDataset + per-task-scale
# normalization against real tujian_v5_recent episodes: real torchcodec video decode
# (now that scripts/vision_install_ffmpeg.sbatch fixed the missing ffmpeg libs) + the
# real per-task-scale table haohw's tacWAM already has on VISION at
# haohw/tacWAM/docs/v10/per_task_scale_normalization/ (byte-identical to the local
# tacWAM checkout's copy -- no transfer needed).
#
# CPU-only: torchcodec's default decode is CPU/software, this is a correctness +
# throughput smoke check, not a GPU benchmark. Restricted to one date subdir (not the
# whole 2.5TB corpus) to keep this a genuine smoke test, not a full-corpus index.
#
# Submit with: sbatch scripts/vision_smoke_test_online_loader.sbatch
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

RAW_ROOT=/scratch/project/prj-02-phai-lab/haohw/tacWAM/data/tujian_v5_recent
NORM=/scratch/project/prj-02-phai-lab/haohw/tacWAM/docs/v10/per_task_scale_normalization/normalization_v10_pad30_per_task_scale.json

# ${SMOKE_DATE:-itw07-23}: overridable via --export=ALL,SMOKE_DATE=itw08-10 so the same
# job definition can re-check coverage on a date inside the per-task-scale table's own
# fit range (docs/v10/manifests/recording_roots_v7_post0802.json covers itw08-03..
# itw08-31 only -- itw07-23 legitimately has 0% task_scale coverage because it's simply
# outside that range, not a code bug; confirmed via job 534584).
python scripts/smoke_test_online_loader.py "$RAW_ROOT" "$NORM" \
  --dates "${SMOKE_DATE:-itw07-23}" --max-episodes 15 --samples-per-episode 3
