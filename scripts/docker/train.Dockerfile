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

# Anaconda now gates the default `main`/`r` channels behind a Terms of Service
# click-through; a fresh install's non-interactive `conda create` fails with
# CondaToSNonInteractiveError until it's accepted explicitly (hit on lab 2026-09-15 --
# serve_policy.Dockerfile predates this gate, hence no mention there). The `conda tos
# accept` calls below must come BEFORE `conda create`, not folded into a comment mid-chain
# (a `#`-comment line inside a `\`-continued RUN breaks the `&&` chain in `sh`/`bash`).
ARG MINICONDA_VERSION=py311_25.5.1-0
RUN curl -fsSL \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
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

# requirements.txt pins a bare `torch==2.7.1` (no CUDA-variant suffix), which PyPI's
# default index resolves to a cu126-class wheel -- that only supports up to sm_90 and
# hard-fails ("no kernel image is available for execution on the device") on newer
# GPUs (confirmed on lab's RTX PRO 6000 Blackwell, sm_120, 2026-09-15; matches this
# project's own established convention that ANY pip-installed torch on Blackwell needs
# a cu128+ wheel -- VISION's env already does this).
#
# This MUST run AFTER `pip install -r requirements.txt`, not before: an earlier attempt
# pre-installed cu128 torch first, expecting the later requirements.txt install to see
# an already-satisfied exact-version pin and leave it alone -- instead, `lerobot`'s own
# (unconstrained) transitive `torch` dependency made pip silently UNINSTALL the cu128
# build and reinstall plain cu126 from PyPI mid-resolution (confirmed in the build log:
# "Found existing installation: torch 2.7.1+cu128 / Uninstalling... / Successfully
# installed ... torch-2.7.1"). Reinstalling cu128 LAST, with --force-reinstall --no-deps
# (skip re-pulling torch's own already-satisfied deps like numpy), guarantees the final
# image state regardless of what requirements.txt's resolution did in between. cu128
# wheels remain compatible with older (Ampere/Hopper) architectures too given a
# reasonably current driver, so this is a safe default, not Blackwell-only -- override
# via --build-arg TORCH_INDEX_URL=... only if a target truly needs something older.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --force-reinstall --no-deps \
        torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
        --index-url "$TORCH_INDEX_URL" \
    && python -c "import torch; assert torch.version.cuda and torch.version.cuda.startswith('12.8'), torch.version.cuda"

# awscli: lets scripts/docker/run_stage1.sh / run_posttrain.sh pull an operator's own
# S3 data with nothing else installed on the host besides the image + their AWS
# credentials (mounted in at `docker run` time, e.g. `-v ~/.aws:/root/.aws:ro`) -- this
# image never bakes in any credentials itself.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install awscli

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
# `.git/` is deliberately NOT excluded by .dockerignore (unlike other build-context
# trimming there) specifically so this COPY brings in a real, remote-configured git
# checkout -- an operator can `git pull` inside a running container to pick up a code
# update without rebuilding the whole image (requires network + read access to the
# repo at pull time, same as any other git pull).
COPY . /app
RUN python -m pip install -e . --no-deps

# No ENTRYPOINT on purpose: every documented invocation (scripts/docker/README.md,
# run_stage1.sh, run_posttrain.sh) already passes a full command, e.g.
# `docker run ... n0vtla_train bash train_stage1.sh`. An `ENTRYPOINT ["/bin/bash"]`
# here would make Docker exec `/bin/bash bash train_stage1.sh` for that exact command
# -- bash resolves the literal arg `bash` via PATH to the real `/usr/bin/bash` binary
# and tries to source IT as a script, failing with a confusing "cannot execute binary
# file" (hit on lab 2026-09-15, initially looked like image corruption, wasn't).
