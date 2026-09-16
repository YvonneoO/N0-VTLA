#!/usr/bin/env bash
set -euo pipefail

# Downloads everything needed to start directly at Stage-2 (skips running Stage-1
# yourself -- uses our own already-trained Stage-1 checkpoint instead). OpenNeoData and
# our checkpoint dataset are both gated; HF_TOKEN is baked into the image, not something
# you need to provide.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Always use the token baked into the image at build time, regardless of whatever
# HF_TOKEN happens to be set to at `docker run` time -- an already-sent older copy of
# open_container.sh still asks the user to fill in their own HF_TOKEN placeholder and
# passes it via `-e`, which would otherwise silently override/break this one.
if [[ -s /opt/hf_token ]]; then
  export HF_TOKEN="$(cat /opt/hf_token)"
fi

STAGE1_STEP="${STAGE1_STEP:-14000}"
# Unset/empty -> download_openneodata_sample.py's own default (all 7 OpenNeoData platforms).
# Set to a comma-separated subset (e.g. "flexiv,umi") to narrow it.
OPENNEODATA_PLATFORMS="${OPENNEODATA_PLATFORMS:-}"
OPENNEODATA_TARGET_PCT="${OPENNEODATA_TARGET_PCT:-5}"
OPENNEODATA_MAX_GB="${OPENNEODATA_MAX_GB:-470}"
OPENNEODATA_DIR="data/openneodata_sample"

echo "[1/4] base checkpoint"
hf download NeoteAI/n0-vtla-base --local-dir checkpoints/n0-vtla-base

echo "[2/4] our Stage-1 checkpoint (step $STAGE1_STEP)"
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0-vtla_ts_pretrain/$STAGE1_STEP/model.safetensors" --local-dir checkpoints

echo "[3/4] Stage-2 data slice (${OPENNEODATA_PLATFORMS:-all platforms}, ${OPENNEODATA_TARGET_PCT}% / max ${OPENNEODATA_MAX_GB}GB) + norm stats"
DOWNLOAD_ARGS=(--target-pct "$OPENNEODATA_TARGET_PCT" --max-gb "$OPENNEODATA_MAX_GB" --output "$OPENNEODATA_DIR")
[[ -n "$OPENNEODATA_PLATFORMS" ]] && DOWNLOAD_ARGS+=(--platforms "$OPENNEODATA_PLATFORMS")
python scripts/download_openneodata_sample.py "${DOWNLOAD_ARGS[@]}"

# One platform subdir per platform actually written (a platform can be skipped if its
# proportional share of the target rounds to zero files) -- discover them rather than
# assuming all requested platforms landed. Saved to a file so run_stage2_onward.sh (a
# separate `bash` process, doesn't inherit these shell variables) can reconstruct the
# same list without re-deriving it.
mapfile -t PLATFORM_DIRS < <(find "$OPENNEODATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#PLATFORM_DIRS[@]} -eq 0 ]]; then
  echo "no OpenNeoData platform directories were written under $OPENNEODATA_DIR" >&2
  exit 1
fi
printf '%s\n' "${PLATFORM_DIRS[@]}" > "$OPENNEODATA_DIR/.platforms"
echo "platforms written: ${PLATFORM_DIRS[*]}"

NORM_ARGS=()
for p in "${PLATFORM_DIRS[@]}"; do
  NORM_ARGS+=(--repo-id "$OPENNEODATA_DIR/$p")
  case "$p" in
    umi|arx5|aloha) NORM_ARGS+=(--robot aloha) ;;  # bimanual platforms
    *) NORM_ARGS+=(--robot flexiv) ;;              # single-arm platforms
  esac
done
python scripts/compute_canonical_norm.py --train-config-name vtla_stage2_align_expert \
  "${NORM_ARGS[@]}" --asset-id openneodata_sample

echo "[4/4] post-train wetlab data (ready-made) + norm stats"
# NOTE: --include takes multiple space-separated patterns in ONE flag (nargs='*');
# passing --include twice makes the second occurrence silently replace the first.
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0vtla_wetlab_canonical_v2/train/**" "n0vtla_wetlab_canonical_v2/holdout/**" \
  --local-dir data
python scripts/compute_canonical_norm.py --train-config-name vtla_tactile_posttrain --robot aloha \
  --repo-id data/n0vtla_wetlab_canonical_v2/train --asset-id wetlab_v2_train

echo "prereqs ready. Run scripts/docker/run_stage2_onward.sh next."
