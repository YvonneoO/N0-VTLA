#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for an operator who only has the image + their own AWS
# credentials: syncs raw itw data from S3, then launches train_stage1.sh -- one
# command instead of a manual `aws s3 sync` + separate train_stage1.sh invocation.
# Docker-specific (assumes `aws` CLI, only guaranteed inside this image) -- unlike
# train_stage1.sh itself, which works with or without Docker.
#
# Usage:
#   docker run --rm --gpus=all \
#     -v ~/.aws:/root/.aws:ro -v $PWD/data:/data -v $PWD/checkpoints:/app/checkpoints \
#     -e VTLA_PRETRAINED_CHECKPOINT=/app/checkpoints/n0-vtla-base \
#     n0vtla_train bash scripts/docker/run_stage1.sh s3://bucket/prefix/itw_raw
#
# AWS credentials/config are never baked into the image -- mount them in
# (`-v ~/.aws:/root/.aws:ro`) or pass AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/etc. as
# `-e` at `docker run` time.

if [[ $# -lt 1 || "$1" != s3://* ]]; then
  cat <<'EOF' >&2
Usage: bash scripts/docker/run_stage1.sh s3://bucket/prefix [extra train_stage1.sh args...]

Env:
  VTLA_ITW_RAW_ROOT_LOCAL=/data/itw_raw   where the S3 data lands (default shown)
  (all of train_stage1.sh's own env vars still apply, e.g. VTLA_PRETRAINED_CHECKPOINT)
EOF
  exit 1
fi
S3_URI="$1"
shift
LOCAL_DIR="${VTLA_ITW_RAW_ROOT_LOCAL:-/data/itw_raw}"

echo "Syncing $S3_URI -> $LOCAL_DIR ..."
mkdir -p "$LOCAL_DIR"
aws s3 sync "$S3_URI" "$LOCAL_DIR"

export VTLA_ITW_RAW_ROOT="$LOCAL_DIR"
exec bash train_stage1.sh "$@"
