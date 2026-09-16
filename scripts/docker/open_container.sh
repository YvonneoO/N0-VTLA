#!/usr/bin/env bash
set -euo pipefail

# Pulls the image and opens a shell inside it, with local dirs mounted for
# data/checkpoints/assets so downloads and training outputs persist on the host.
#
# EDIT THESE THREE PLACEHOLDERS before running:
DATA_DIR="/path/to/your/data"
CHECKPOINTS_DIR="/path/to/your/checkpoints"
ASSETS_DIR="/path/to/your/assets"

# HF_TOKEN is already baked into the image (OpenNeoData + our checkpoint dataset are
# gated) -- you do NOT need to set this. Only set it if you want to override with your
# own token for some reason; if HF_TOKEN is unset/empty here, the image's own baked-in
# default is left untouched at runtime.
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$DATA_DIR" "$CHECKPOINTS_DIR" "$ASSETS_DIR"

docker pull qqyang/n0vtla_train:latest

# --shm-size: PyTorch's DataLoader workers pass data between processes via /dev/shm;
# Docker's tiny 64MB default crashes with "Bus error (SIGBUS)" once the dataset/batch
# size gets real (hit this ourselves loading real data). 16g is a safe default; raise it
# if you still see SIGBUS.
SHM_SIZE="${SHM_SIZE:-16g}"

# If multi-GPU training crashes with an "NCCL ... illegal memory access" error (a
# P2P/GPU-compatibility issue on some machines, not a code bug), uncomment this line:
# NCCL_P2P_ARGS=(-e NCCL_P2P_DISABLE=1)

# Only override the image's baked-in HF_TOKEN if the caller set their own -- an empty
# `-e HF_TOKEN=` would otherwise clobber the baked-in default with an empty string.
HF_TOKEN_ARGS=()
[[ -n "$HF_TOKEN" ]] && HF_TOKEN_ARGS=(-e HF_TOKEN="$HF_TOKEN")

docker run --rm --gpus=all -it --shm-size="$SHM_SIZE" \
  -v "$DATA_DIR":/app/data \
  -v "$CHECKPOINTS_DIR":/app/checkpoints \
  -v "$ASSETS_DIR":/app/assets \
  "${HF_TOKEN_ARGS[@]:-}" \
  "${NCCL_P2P_ARGS[@]:-}" \
  qqyang/n0vtla_train:latest bash
