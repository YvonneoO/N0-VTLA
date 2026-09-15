#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for an operator who only has the image + their own AWS
# credentials: syncs a LeRobot-format robot dataset from S3, then launches train.sh --
# one command instead of a manual `aws s3 sync` + separate train.sh invocation.
# Docker-specific (assumes `aws` CLI, only guaranteed inside this image) -- unlike
# train.sh itself, which works with or without Docker.
#
# Usage:
#   docker run --rm --gpus=all \
#     -v ~/.aws:/root/.aws:ro -v $PWD/data:/data -v $PWD/checkpoints:/app/checkpoints \
#     -v $PWD/assets:/app/assets \
#     -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
#     -e VTLA_ASSET_ID=my_dataset \
#     n0vtla_train bash scripts/docker/run_posttrain.sh s3://bucket/prefix/robot_dataset
#
# AWS credentials/config are never baked into the image -- mount them in
# (`-v ~/.aws:/root/.aws:ro`) or pass AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/etc. as
# `-e` at `docker run` time. A precomputed norm_stats.json under
# assets/<config_name>/<asset_id>/ is still the operator's own responsibility (see
# scripts/docker/README.md) -- it's not part of the raw dataset sync below.

if [[ $# -lt 1 || "$1" != s3://* ]]; then
  cat <<'EOF' >&2
Usage: bash scripts/docker/run_posttrain.sh s3://bucket/prefix [extra train.sh args...]

Env:
  VTLA_DATASET_PATH_LOCAL=/data/robot_dataset   where the S3 data lands (default shown)
  (all of train.sh's own env vars still apply, e.g. VTLA_PRETRAINED_CHECKPOINT, VTLA_ASSET_ID)
EOF
  exit 1
fi
S3_URI="$1"
shift
LOCAL_DIR="${VTLA_DATASET_PATH_LOCAL:-/data/robot_dataset}"

echo "Syncing $S3_URI -> $LOCAL_DIR ..."
mkdir -p "$LOCAL_DIR"
aws s3 sync "$S3_URI" "$LOCAL_DIR"

export VTLA_DATASET_PATH="$LOCAL_DIR"
exec bash train.sh "$@"
