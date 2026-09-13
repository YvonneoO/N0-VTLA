#!/bin/bash
#SBATCH --job-name=n0vtla-install-ffmpeg
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/N0-VTLA/install_ffmpeg_%j.log
#
# torchcodec (used by n0vtla/training/itw_online_dataset.py for random-access video
# decode) needs FFmpeg's shared libraries (libavutil/libavformat/...) at runtime --
# it does NOT bundle them. The n0vtla conda env has none of FFmpeg 4/5/6/7 installed,
# so torchcodec.decoders.VideoDecoder fails with "Could not load libtorchcodec"
# (confirmed on job 534558's log). conda-forge's ffmpeg package ships those .so files
# directly into the env's lib/ dir, which is exactly what torchcodec's loader searches.
#
# Submit with: sbatch scripts/vision_install_ffmpeg.sbatch  (run from the repo root)
set -euo pipefail

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/project/prj-02-phai-lab/yqq/envs/n0vtla

conda install -y -c conda-forge "ffmpeg<8"

python -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec import OK:', torchcodec.__version__)"
echo VISION_FFMPEG_INSTALL_COMPLETE
