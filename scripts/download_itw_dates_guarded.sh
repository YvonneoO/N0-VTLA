#!/usr/bin/env bash
set -euo pipefail

BUCKET="${BUCKET:-noitom-us-west-2}"
REGION="${REGION:-us-west-2}"
DEST_ROOT="${DEST_ROOT:-/DATA2/qianqian/n0vtla_itw_raw}"
LOG_DIR="${LOG_DIR:-$DEST_ROOT/logs}"
MAX_GB="${MAX_GB:-300}"
EXCLUDE_DEPTH="${EXCLUDE_DEPTH:-1}"

mkdir -p "$DEST_ROOT" "$LOG_DIR"

bytes_used() {
  du -sb "$DEST_ROOT" 2>/dev/null | awk '{print $1}'
}

max_bytes=$((MAX_GB * 1024 * 1024 * 1024))
dates=(
  itw08-03
  itw08-06
  itw08-07
  itw08-10
  itw08-11
  itw08-12
  itw08-13
  itw08-14
  itw08-15
  itw08-17
)

for date in "${dates[@]}"; do
  used="$(bytes_used)"
  if (( used >= max_bytes )); then
    echo "$(date '+%F %T') reached ${MAX_GB}GB guard before $date; stopping"
    exit 0
  fi

  echo "$(date '+%F %T') syncing s3://$BUCKET/$date/ -> $DEST_ROOT/$date/"
  args=(s3 sync "s3://$BUCKET/$date/" "$DEST_ROOT/$date/" --region "$REGION" --only-show-errors)
  if [[ "$EXCLUDE_DEPTH" == "1" ]]; then
    args+=(--exclude "depth_head.mkv")
  fi
  aws "${args[@]}" 2>&1 | tee -a "$LOG_DIR/sync_${date}.log"

  used="$(bytes_used)"
  echo "$(date '+%F %T') completed $date; local raw size=$(( used / 1024 / 1024 / 1024 )) GiB"
done

echo "$(date '+%F %T') completed requested date range"
