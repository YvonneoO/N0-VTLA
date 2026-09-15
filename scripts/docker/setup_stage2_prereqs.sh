#!/usr/bin/env bash
set -euo pipefail

# Downloads everything needed to start directly at Stage-2 (skips running Stage-1
# yourself -- uses our own already-trained Stage-1 checkpoint instead). Needs
# HF_TOKEN (OpenNeoData is gated; everything else here is public).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

STAGE1_STEP="${STAGE1_STEP:-14000}"
OPENNEODATA_PLATFORM="${OPENNEODATA_PLATFORM:-flexiv}"
OPENNEODATA_EPISODES="${OPENNEODATA_EPISODES:-2}"

echo "[1/4] base checkpoint"
hf download NeoteAI/n0-vtla-base --local-dir checkpoints/n0-vtla-base

echo "[2/4] our Stage-1 checkpoint (step $STAGE1_STEP)"
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0-vtla_ts_pretrain/$STAGE1_STEP/model.safetensors" --local-dir checkpoints

echo "[3/4] Stage-2 data slice ($OPENNEODATA_PLATFORM, $OPENNEODATA_EPISODES episodes) + norm stats"
python scripts/download_openneodata_flexiv_smoke.py --platform "$OPENNEODATA_PLATFORM" \
  --output data/openneodata_smoke --num-episodes "$OPENNEODATA_EPISODES"
python scripts/compute_canonical_norm.py --train-config-name vtla_stage2_align_expert \
  --repo-id data/openneodata_smoke --asset-id openneodata_smoke

echo "[4/4] post-train wetlab data (ready-made) + norm stats"
hf download qqyang/zihiao_real_test --repo-type dataset \
  --include "n0vtla_wetlab_canonical_v2/train/**" \
  --include "n0vtla_wetlab_canonical_v2/holdout/**" --local-dir data
python scripts/compute_canonical_norm.py --train-config-name vtla_tactile_posttrain --robot aloha \
  --repo-id data/n0vtla_wetlab_canonical_v2/train --asset-id wetlab_v2_train

echo "prereqs ready. Run scripts/docker/run_stage2_onward.sh next."
