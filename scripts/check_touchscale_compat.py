"""Compatibility of the TouchScale corpus (touchscale_data/) with the Stage-1 online loader and the old splits.
1. old scaling manifest (docs/artifacts/itw_scaling_splits.json): how many of its episodes exist here (by date/uuid)
2. date-range subsets: episodes/hours for the old 100% date range (itw07-23..itw09-10) and for all dates
3. glove units/values vs the per-task-scale normalization; task-name coverage of its task_scale table
4. aligned_timeline_h5 on a random sample: pass rate and failure reasons
Usage: python scripts/check_touchscale_compat.py ROOT NORM_JSON MANIFEST_JSON"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itw_pressure as ip  # noqa: E402

root, norm_path, man_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
idx = pd.read_csv(root / "index/episode_index.csv", low_memory=False)
use = idx[idx.complete & idx.idx_ok.eq(True)].copy()
use["key"] = use.day + "/" + use.uuid
have = set(use.key)
man = json.loads(man_path.read_text())
print("==1 manifest coverage (usable episodes in touchscale_data)")
for k, v in man["splits"].items():
    print(f"  {k}: {sum(1 for p in v if p in have)}/{len(v)}")
mdays = sorted({p.split('/')[0] for v in man["splits"].values() for p in v})
print("  manifest dates", len(mdays), mdays[0], "..", mdays[-1], "| missing here:", [d for d in mdays if d not in set(use.day)])
print("==2 date-range subsets")
for name, lo, hi in (("old100_itw0723_0910", "itw07-23", "itw09-10"), ("manifest_range_0723_0915", "itw07-23", "itw09-15"), ("all", "itw00-00", "itw99-99")):
    s = use[(use.day >= lo) & (use.day <= hi)]
    print(f"  {name}: {len(s)} eps, {s.duration_s.sum() / 3600:.1f} h, {s.day.nunique()} days")
print("==3 units / values / task coverage")
norm = ip.load_normalization(norm_path)
rng = random.Random(0)
sample = use.sample(200, random_state=0)
ep = root / sample.iloc[0].rel_dir
with h5py.File(ep / "episode.h5", "r") as f:
    print("  attrs left:", dict(f["obs/tactile/raw/left"].attrs))
    print("  metadata:", {k: str(v)[:60] for k, v in f["metadata"].attrs.items()})
vals = {h: [] for h in ip.HANDS}
for _, r in sample.head(40).iterrows():
    with h5py.File(root / r.rel_dir / "episode.h5", "r") as f:
        for h in ip.HANDS:
            vals[h].append(np.concatenate([f[f"obs/tactile/raw/{h}/pad_{p}"][:].reshape(-1) for p in ip.PAD_IDS]))
for i, h in enumerate(ip.HANDS):
    v = np.concatenate(vals[h])
    b = np.array(norm["normal_baseline"][i * 15:(i + 1) * 15])
    print(f"  {h}: raw p5 {np.percentile(v,5):.4g} p50 {np.median(v):.4g} p99 {np.percentile(v,99):.4g} p99.9 {np.percentile(v,99.9):.4g} | norm baseline range {b.min():.4g}..{b.max():.4g}")
if "task_scale" in norm:
    table = norm["task_scale"]
    print(f"  task_scale entries {len(table)}, default_scale {norm.get('default_scale')}, sample scales {list(table.values())[:5]}")
    m_all = use.task_name.isin(table)
    print(f"  task-name coverage: all {m_all.mean():.1%} of eps; old100 range {m_all[(use.day <= 'itw09-10')].mean():.1%}; after itw09-10 {m_all[(use.day > 'itw09-10')].mean():.1%}")
print("==4 aligned_timeline_h5 on 200 random usable episodes")
ok, reasons, lens = 0, Counter(), []
for _, r in sample.iterrows():
    try:
        m, _a = ip.aligned_timeline_h5(root / r.rel_dir)
        ok += 1
        lens.append(len(m["master_timestamp_ns"]) / max(r.n_frames, 1))
    except Exception as e:  # noqa: BLE001
        reasons[str(e).split(":")[-1].strip()[:70]] += 1
print(f"  pass {ok}/200; aligned frames / index n_frames median {np.median(lens):.3f}; failures {dict(reasons.most_common(5))}")
print("COMPAT_DONE")
