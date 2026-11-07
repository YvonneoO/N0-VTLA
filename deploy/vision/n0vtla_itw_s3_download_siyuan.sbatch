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
#   1. size estimate from `aws s3 ls --recursive` per date: total, without depth video, per file kind (ITS| lines)
#   2. MODE=download (default) and the estimate (without depth) <= MAX_GIB: `aws s3 sync` per date, excluding
#      every depth file (*depth*); restart-safe (sync skips files already here).
# Env: MODE (estimate | download), DATES (default: the manifest's dates), MAX_GIB (default 2500), DEST.
#   sbatch --export=ALL,MODE=estimate deploy/vision/n0vtla_itw_s3_download_siyuan.sbatch
set -uo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
AWS=$YQQ/envs/tactile/bin/aws
BUCKET=s3://noitom-us-west-2
DEST="${DEST:-$YQQ/data/raw}"
MODE="${MODE:-download}"; MAX_GIB="${MAX_GIB:-2500}"
MANIFEST=$YQQ/data/itw_scaling_splits.json
OUT=$YQQ/logs/itw_s3_$SLURM_JOB_ID; mkdir -p "$OUT" "$DEST"
[[ -n "${AWS_SHARED_CREDENTIALS_FILE:-}" && -f "$AWS_SHARED_CREDENTIALS_FILE" ]] || { echo "ITS|fatal: no AWS credentials file"; exit 1; }
if [[ -z "${DATES:-}" ]]; then
  DATES=$($YQQ/envs/n0vtla/bin/python -c "import json; d=json.load(open('$MANIFEST')); print(' '.join(sorted({p.split('/')[0] for v in d['splits'].values() for p in v})))")
fi
read -r -a dates <<< "$DATES"
echo "ITS|${#dates[@]} dates: ${dates[0]} .. ${dates[-1]}  mode=$MODE max_gib=$MAX_GIB dest=$DEST"

# ---- 1. estimate
for d in "${dates[@]}"; do
  "$AWS" s3 ls "$BUCKET/$d/" --recursive > "$OUT/ls_$d.txt" 2> "$OUT/ls_$d.err" || echo "ITS|ls failed for $d: $(head -c 200 "$OUT/ls_$d.err")"
done
$YQQ/envs/n0vtla/bin/python - "$OUT" "${dates[@]}" <<'EOF'
import sys, collections, pathlib
out, dates = pathlib.Path(sys.argv[1]), sys.argv[2:]
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
        if "depth" not in key: nd += size
        segs = key.split("/")
        if len(segs) >= 3: eps.add(segs[1])
    rows.append((d, len(eps), t / G, nd / G)); tot += t; nodepth += nd; eps_all |= {(d, e) for e in eps}
for d, n, t, nd in rows:
    print(f"ITS|{d}: {n} episodes, {t:.1f} GiB all, {nd:.1f} GiB without depth", flush=True)
print(f"ITS|TOTAL {len(eps_all)} episodes: {tot / G:.1f} GiB all, {nodepth / G:.1f} GiB without depth", flush=True)
for k, b in kind_bytes.most_common(14):
    print(f"ITS|kind {k}: {kind_n[k]} files, {b / G:.1f} GiB{'  (excluded)' if 'depth' in k else ''}", flush=True)
(out / "nodepth_gib.txt").write_text(f"{nodepth / G:.0f}\n")
EOF
NEED=$(cat "$OUT/nodepth_gib.txt" 2>/dev/null || echo 999999)
[[ "$MODE" == "download" ]] || { echo "ITS|estimate only; done"; exit 0; }
if (( NEED > MAX_GIB )); then echo "ITS|need ${NEED} GiB > MAX_GIB=${MAX_GIB}; not downloading"; exit 0; fi

# ---- 2. download (sync, no depth)
for d in "${dates[@]}"; do
  echo "ITS|$(date +%H:%M:%S) sync $d"
  "$AWS" s3 sync "$BUCKET/$d/" "$DEST/$d/" --exclude "*depth*" --only-show-errors > "$OUT/sync_$d.log" 2>&1 \
    || echo "ITS|sync $d exit $? (see $OUT/sync_$d.log)"
  echo "ITS|$(date +%H:%M:%S) done $d: $(ls -1 "$DEST/$d" | wc -l) episode dirs, $(du -s --block-size=1G "$DEST/$d" | cut -f1) GiB"
done
echo "ITS|raw tree: $(du -s --block-size=1G "$DEST" | cut -f1) GiB"
echo "ITW_S3_DONE"
