# Dockerfile for Stage-1 (human ITW tactile, action-free) and post-train (robot
# LeRobot-format, action-conditioned) training. Stage-2/3 are NOT included (separate
# branch, not ready). Adapted from serve_policy.Dockerfile's proven base-image/conda/
# transformers-patch skeleton -- see that file for the serving-only equivalent.
#
# Build the container:
#   docker build . -t n0vtla_train -f scripts/docker/train.Dockerfile
#   # Different GPU architecture (e.g. Blackwell/cu128)? Override the base image tag:
#   docker build . -t n0vtla_train -f scripts/docker/train.Dockerfile \
#     --build-arg CUDA_IMAGE_TAG=12.8.0-cudnn-runtime-ubuntu22.04
#
# Run: see scripts/docker/README.md for full launch commands + required volume mounts
# (this image is self-contained code-wise, but data/checkpoints are runtime mounts,
# not baked in -- see README for why).

# Not digest-pinned (unlike serve_policy.Dockerfile) because the target server's GPU
# architecture is unknown at build time -- callers override via --build-arg for
# anything other than the common Ampere/Hopper case this default targets.
ARG CUDA_IMAGE_TAG=12.2.2-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        git \
        git-lfs \
        linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

ARG MINICONDA_VERSION=py311_25.5.1-0
RUN curl -fsSL \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda create -y -n vtla python=3.11 \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/vtla/bin:/opt/conda/bin:$PATH
ENV PYTHONPATH=/app

# NEW vs. serve_policy.Dockerfile: Stage-1's online loader decodes raw itw .mp4s via
# torchcodec.decoders.VideoDecoder, which needs FFmpeg's shared libs at runtime -- the
# torchcodec pip wheel does NOT bundle them (real failure hit on VISION: "Could not
# load libtorchcodec", fixed the same way there -- see scripts/vision_install_ffmpeg.sbatch).
# serve_policy.Dockerfile never decodes raw video, so it never needed this.
RUN /opt/conda/bin/conda install -y -n vtla -c conda-forge "ffmpeg<8" \
    && /opt/conda/bin/conda clean -afy

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    GIT_LFS_SKIP_SMUDGE=1 python -m pip install -r /tmp/requirements.txt
# Note: requirements.txt pins BOTH opencv-python and opencv-python-headless -- if a
# `cv2` import ever breaks post-install, check for a conflict between these two here
# first (not fixed in this Dockerfile; the pins are deliberate upstream, not touched).

# Copy transformers_replace files while preserving directory structure (MANDATORY --
# n0vtla depends on a patched transformers, not the stock pip package).
COPY n0vtla/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

# Bake DINOv2 in at build time (network needed here only) so runtime stays fully
# offline per train.sh/train_stage1.sh's HF_HUB_OFFLINE=1 default -- small (~90M
# params), identical every run, removes a first-run-download flakiness class on a
# locked-down external server.
ENV HF_HOME=/opt/hf_cache
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
    "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"

# Self-contained image: the code is copied in, not bind-mounted (unlike
# compose.yml's serving setup) -- runtime data (raw itw episodes, LeRobot dataset,
# pretrained checkpoint) stays mounted as volumes; only the code+deps are baked in.
# The committed assets/itw_normalization/ JSON comes along automatically here.
COPY . /app
RUN python -m pip install -e . --no-deps

ENTRYPOINT ["/bin/bash"]
