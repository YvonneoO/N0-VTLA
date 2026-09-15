#!/usr/bin/env bash
set -euo pipefail

# Run this ONE script inside the container (after open_container.sh) to go from
# nothing to a post-trained checkpoint: downloads prereqs, then Stage-2 -> merge
# -> post-train. See setup_stage2_prereqs.sh / run_stage2_onward.sh for the
# individual steps and their env var overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/setup_stage2_prereqs.sh"
bash "$SCRIPT_DIR/run_stage2_onward.sh" "$@"
