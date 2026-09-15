#!/usr/bin/env bash
set -euo pipefail

# Pulls the image and opens a shell inside it, with local dirs mounted for
# data/checkpoints/assets so downloads and training outputs persist on the host.
#
# EDIT THESE FOUR PLACEHOLDERS before running:
DATA_DIR="/path/to/your/data"
CHECKPOINTS_DIR="/path/to/your/checkpoints"
ASSETS_DIR="/path/to/your/assets"
HF_TOKEN="<your Hugging Face token, needs OpenNeoData gated-dataset access>"

mkdir -p "$DATA_DIR" "$CHECKPOINTS_DIR" "$ASSETS_DIR"

docker pull qqyang/n0vtla_train:latest

docker run --rm --gpus=all -it \
  -v "$DATA_DIR":/app/data \
  -v "$CHECKPOINTS_DIR":/app/checkpoints \
  -v "$ASSETS_DIR":/app/assets \
  -e HF_TOKEN="$HF_TOKEN" \
  qqyang/n0vtla_train:latest bash
