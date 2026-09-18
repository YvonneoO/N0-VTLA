#!/bin/bash
#SBATCH --job-name=n0vtla-env-fix-ffmpeg-siyuan
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/env_fix_ffmpeg_%j.log
#
# Fixes the siyuan account's n0vtla conda env: the 2-GPU smoke test (job 560914)
# crashed with "Could not load libtorchcodec ... libavutil.so.* cannot open shared
# object file" for FFmpeg versions 4 through 7 -- FFmpeg is missing from this env
# entirely. The online ITW dataloader decodes video via torchcodec, so this is a
# hard requirement, not optional (see docs/README.md's "Could not load
# libtorchcodec" troubleshooting entry -- there it's a non-issue because FFmpeg
# ships baked into the Docker image; VISION has no containers, so this env just
# never got it installed).
#
# Installs ffmpeg from conda-forge directly into the env prefix (no `conda
# activate`/`source` needed).
set -euo pipefail

module load Miniforge3
conda install -p /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla -c conda-forge ffmpeg -y
conda list -p /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla ffmpeg

echo N0VTLA_ENV_FIX_FFMPEG_DONE
