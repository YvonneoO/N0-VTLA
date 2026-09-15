#!/usr/bin/env bash
set -euo pipefail

# Downloads full ITW dates from S3, guarded by a size cap (stops before exceeding
# MAX_GB of local disk, does not resume/continue past that point automatically --
# rerun after freeing space or raising MAX_GB). Needs AWS credentials for the
# noitom-us-west-2 bucket (e.g. mount them if running via the n0vtla_train image:
# `-v /path/to/aws/creds:/root/.aws:ro`).
#
# By default, downloads EVERY itwMM-DD-shaped date prefix in the bucket (discovered
# live via `aws s3 ls`, not hardcoded -- new dates show up automatically). The
# bucket also holds a few unrelated prefixes (0706_task/, 20260813/, ITW_001/,
# nus-kitchen-2026/, visualization/) that are NOT part of this convention and are
# always skipped.
#
# Usage examples:
#   # everything except one date (e.g. an excluded pilot batch):
#   EXCLUDE_DATES=itw07-16 bash scripts/download_itw_dates_guarded.sh
#
#   # only specific dates (e.g. for a data-scaling experiment's smaller cut):
#   ONLY_DATES="itw07-23 itw07-25 itw07-27" bash scripts/download_itw_dates_guarded.sh
#
#   # raise the size guard for a real full download (default is a conservative 300GB):
#   MAX_GB=3000 EXCLUDE_DATES=itw07-16 bash scripts/download_itw_dates_guarded.sh

BUCKET="${BUCKET:-noitom-us-west-2}"
REGION="${REGION:-us-west-2}"
DEST_ROOT="${DEST_ROOT:-/DATA2/qianqian/n0vtla_itw_raw}"
LOG_DIR="${LOG_DIR:-$DEST_ROOT/logs}"
MAX_GB="${MAX_GB:-300}"
EXCLUDE_DEPTH="${EXCLUDE_DEPTH:-1}"
ONLY_DATES="${ONLY_DATES:-}"
EXCLUDE_DATES="${EXCLUDE_DATES:-}"

mkdir -p "$DEST_ROOT" "$LOG_DIR"

bytes_used() {
  du -sb "$DEST_ROOT" 2>/dev/null | awk '{print $1}'
}

max_bytes=$((MAX_GB * 1024 * 1024 * 1024))

if [[ -n "$ONLY_DATES" ]]; then
  read -r -a dates <<< "$ONLY_DATES"
else
  # Discover date prefixes live -- only itwMM-DD-shaped ones.
  mapfile -t all_dates < <(
    aws s3 ls "s3://$BUCKET/" --region "$REGION" \
      | awk '{print $2}' | sed 's:/$::' \
      | grep -E '^itw[0-9]{2}-[0-9]{2}$' | sort
  )
  dates=()
  for d in "${all_dates[@]}"; do
    skip=0
    for ex in $EXCLUDE_DATES; do
      [[ "$d" == "$ex" ]] && skip=1 && break
    done
    (( skip )) || dates+=("$d")
  done
fi

echo "will sync ${#dates[@]} date(s): ${dates[*]}"

for d in "${dates[@]}"; do
  used="$(bytes_used)"
  if (( used >= max_bytes )); then
    echo "$(date '+%F %T') reached ${MAX_GB}GB guard before $d; stopping"
    exit 0
  fi

  echo "$(date '+%F %T') syncing s3://$BUCKET/$d/ -> $DEST_ROOT/$d/"
  args=(s3 sync "s3://$BUCKET/$d/" "$DEST_ROOT/$d/" --region "$REGION" --only-show-errors)
  if [[ "$EXCLUDE_DEPTH" == "1" ]]; then
    args+=(--exclude "depth_head.mkv")
  fi
  aws "${args[@]}" 2>&1 | tee -a "$LOG_DIR/sync_${d}.log"

  used="$(bytes_used)"
  echo "$(date '+%F %T') completed $d; local raw size=$(( used / 1024 / 1024 / 1024 )) GiB"
done

echo "$(date '+%F %T') completed requested date range"
