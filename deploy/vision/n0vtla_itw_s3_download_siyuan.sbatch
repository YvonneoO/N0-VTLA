#!/bin/bash
#SBATCH --job-name=n0vtla-itw-s3
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/itw_s3_%j.log
#
# Raw ITW (tacWAM layout) dates from s3://noitom-us-west-2 -> yqq/data/raw/<date>/ for mid-train on the old 100%
# date set (the 39 dates of docs/artifacts/itw_scaling_splits.json, itw07-23..itw09-15). CPU-only I/O job.
#   1. size estimate from `aws s3 ls --recursive` per date: total, without the EXCLUDE files, per file kind (ITS|)
#   2. MODE=download (default) and the estimate (without EXCLUDE) <= MAX_GIB: `aws s3 sync` per date with those
#      excludes; restart-safe (sync skips files already here). Excluded files already on disk are removed.
# EXCLUDE (space-separated aws/fnmatch patterns, relative to <date>/): default = what the raw-layout loader never
#   reads -- depth video (*depth*), episode.h5 (TouchScale packaging only; the raw layout uses csv + *_hand_data.npz),
#   glove calibration */left.json */right.json, */imu.txt. Kept: {rgb_head,wrist_left,wrist_right}.{mp4,csv},
#   {left,right}_hand_data.npz, task_info.json, dataset.csv.
# Env: MODE (estimate | download), DATES (default: the manifest's dates), MAX_GIB (default 2500), DEST, EXCLUDE.
#   sbatch --export=ALL,MODE=estimate deploy/vision/n0vtla_itw_s3_download_siyuan.sbatch
set -uo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
AWS=$YQQ/envs/tactile/bin/aws
BUCKET=s3://noitom-us-west-2
DEST="${DEST:-$YQQ/data/raw}"
MODE="${MODE:-download}"; MAX_GIB="${MAX_GIB:-2500}"
EXCLUDE="${EXCLUDE:-*depth* */episode.h5 */left.json */right.json */imu.txt}"
read -r -a excl <<< "$EXCLUDE"; EXARGS=(); for e in "${excl[@]}"; do EXARGS+=(--exclude "$e"); done
MANIFEST=$YQQ/data/itw_scaling_splits.json
OUT=$YQQ/logs/itw_s3_$SLURM_JOB_ID; mkdir -p "$OUT" "$DEST"
[[ -n "${AWS_SHARED_CREDENTIALS_FILE:-}" && -f "$AWS_SHARED_CREDENTIALS_FILE" ]] || { echo "ITS|fatal: no AWS credentials file"; exit 1; }
if [[ -z "${DATES:-}" ]]; then
  DATES=$($YQQ/envs/n0vtla/bin/python -c "import json; d=json.load(open('$MANIFEST')); print(' '.join(sorted({p.split('/')[0] for v in d['splits'].values() for p in v})))")
fi
read -r -a dates <<< "$DATES"
echo "ITS|${#dates[@]} dates: ${dates[0]} .. ${dates[-1]}  mode=$MODE max_gib=$MAX_GIB dest=$DEST exclude=[$EXCLUDE]"

# ---- 1. estimate
for d in "${dates[@]}"; do
  "$AWS" s3 ls "$BUCKET/$d/" --recursive > "$OUT/ls_$d.txt" 2> "$OUT/ls_$d.err" || echo "ITS|ls failed for $d: $(head -c 200 "$OUT/ls_$d.err")"
done
$YQQ/envs/n0vtla/bin/python - "$OUT" "$EXCLUDE" "${dates[@]}" <<'EOF'
import sys, collections, pathlib, fnmatch
out, excl, dates = pathlib.Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3:]
G = 1024 ** 3
kind_bytes, kind_n = collections.Counter(), collections.Counter()
tot = nodepth = 0; eps_all = set(); rows = []
for d in dates:
    t = nd = 0; eps = set()
    for line in (out / f"ls_{d}.txt").read_text().splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4: continue
        size, key = int(parts[2]), parts[3]
        name = key.rsplit("/", 1)[-1]
        k = name if name.count(".") else "(noext)"
        kind_bytes[k] += size; kind_n[k] += 1
        t += size
        if not any(fnmatch.fnmatch(key.split("/", 1)[-1], e) for e in excl): nd += size
        segs = key.split("/")
        if len(segs) >= 3: eps.add(segs[1])
    rows.append((d, len(eps), t / G, nd / G)); tot += t; nodepth += nd; eps_all |= {(d, e) for e in eps}
for d, n, t, nd in rows:
    print(f"ITS|{d}: {n} episodes, {t:.1f} GiB all, {nd:.1f} GiB to download", flush=True)
print(f"ITS|TOTAL {len(eps_all)} episodes: {tot / G:.1f} GiB all, {nodepth / G:.1f} GiB to download (excludes {excl})", flush=True)
for k, b in kind_bytes.most_common(14):
    ex = any(fnmatch.fnmatch("x/" + k, e) for e in excl)
    print(f"ITS|kind {k}: {kind_n[k]} files, {b / G:.1f} GiB{'  (excluded)' if ex else ''}", flush=True)
(out / "nodepth_gib.txt").write_text(f"{nodepth / G:.0f}\n")
EOF
NEED=$(cat "$OUT/nodepth_gib.txt" 2>/dev/null || echo 999999)
[[ "$MODE" == "download" ]] || { echo "ITS|estimate only; done"; exit 0; }
if (( NEED > MAX_GIB )); then echo "ITS|need ${NEED} GiB > MAX_GIB=${MAX_GIB}; not downloading"; exit 0; fi

# ---- 2. download (sync with the excludes)
for d in "${dates[@]}"; do
  echo "ITS|$(date +%H:%M:%S) sync $d"
  if [[ -d "$DEST/$d" ]]; then        # excluded files left by an earlier sync (episode dirs are <date>/<uuid>/)
    for e in "${excl[@]}"; do find "$DEST/$d" -mindepth 2 -maxdepth 2 -type f -path "$DEST/$d/$e" -delete; done
  fi
  "$AWS" s3 sync "$BUCKET/$d/" "$DEST/$d/" "${EXARGS[@]}" --only-show-errors > "$OUT/sync_$d.log" 2>&1 \
    || echo "ITS|sync $d exit $? (see $OUT/sync_$d.log)"
  echo "ITS|$(date +%H:%M:%S) done $d: $(ls -1 "$DEST/$d" | wc -l) episode dirs, $(du -s --block-size=1G "$DEST/$d" | cut -f1) GiB"
done
echo "ITS|raw tree: $(du -s --block-size=1G "$DEST" | cut -f1) GiB"
echo "ITW_S3_DONE"
