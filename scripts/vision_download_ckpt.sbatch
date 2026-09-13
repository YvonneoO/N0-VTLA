#!/bin/bash
#SBATCH --job-name=n0vtla-ckpt-dl
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/ckpt_download_%j.log
#
# Download the public n0-vtla-base checkpoint from HF, off the login node. A real
# bash-shebang script, not `sbatch --wrap` -- --wrap runs under /bin/sh (dash), which
# silently breaks `source`/conda activate (see @TAMU/CLAUDE.md's documented incident;
# reproduced here on the first attempt: conda picked up the wrong system Anaconda and
# `hf` was never found, but the wrap's `;`-chained echo still made the job exit 0).
#
# Submit with: sbatch scripts/vision_download_ckpt.sbatch  (run from the repo root)
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla
source /scratch/project/prj-02-phai-lab/yqq/env.sh

hf download NeoteAI/n0-vtla-base --local-dir /scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base

ls -la /scratch/project/prj-02-phai-lab/yqq/N0-VTLA/checkpoints/n0-vtla-base
echo VISION_CKPT_DOWNLOAD_COMPLETE
