#!/usr/bin/env bash
set -euo pipefail

# Downloads a SMALL subset of episodes from ONE ITW date -- for smoke tests, not
# real training. For a full guarded multi-date sync (all episodes, size-capped)
# see download_itw_dates_guarded.sh instead.
#
# Usage:
#   bash scripts/download_itw_episodes_smoke.sh <date, e.g. itw08-03> [num_episodes] [dest_root]
#
# Lands episodes at <dest_root>/<date>/<episode_uuid>/ -- point VTLA_ITW_RAW_ROOT at
# <dest_root> itself, NOT <dest_root>/<date> (the loader scans <root>/<date>/<episode>/,
# so pointing it directly at the date folder fails with "no recognizable episode dirs" --
# this bit a real test run on 2026-09-15, see git history).

BUCKET="${BUCKET:-noitom-us-west-2}"
REGION="${REGION:-us-west-2}"
DATE="${1:?usage: download_itw_episodes_smoke.sh <date e.g. itw08-03> [num_episodes] [dest_root]}"
NUM_EPISODES="${2:-5}"
DEST_ROOT="${3:-/DATA2/qianqian/n0vtla_itw_raw_smoke}"

echo "listing episodes under s3://$BUCKET/$DATE/"
episodes="$(aws s3 ls "s3://$BUCKET/$DATE/" --region "$REGION" \
  | awk '{print $2}' | sed 's:/$::' | grep -v '^$' | head -n "$NUM_EPISODES")"

if [[ -z "$episodes" ]]; then
  echo "no episodes found under s3://$BUCKET/$DATE/" >&2
  exit 1
fi

while IFS= read -r ep; do
  echo "syncing $ep"
  aws s3 sync "s3://$BUCKET/$DATE/$ep/" "$DEST_ROOT/$DATE/$ep/" \
    --region "$REGION" --exclude "depth_head.mkv" --only-show-errors
done <<< "$episodes"

echo "done: $DEST_ROOT/$DATE ($(echo "$episodes" | wc -l) episodes)"
echo "set VTLA_ITW_RAW_ROOT=$DEST_ROOT (the parent of $DATE, not $DEST_ROOT/$DATE)"
