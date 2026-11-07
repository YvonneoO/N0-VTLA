"""Describe a TouchScale corpus tree (touchscale_data/): H5 layout of one episode, index statistics, README.
Usage: python scripts/inspect_touchscale_corpus.py ROOT"""
import sys
from pathlib import Path

import h5py
import pandas as pd

root = Path(sys.argv[1])
print("==README (tail)")
print("\n".join((root / "index/README.md").read_text().splitlines()[20:67]))
idx = pd.read_csv(root / "index/episode_index.csv", low_memory=False)
use = idx[idx.complete & idx.idx_ok.eq(True)]
print("==INDEX rows", len(idx), "usable", len(use), "hours usable %.1f all %.1f" % (use.duration_s.sum() / 3600, idx.duration_s.sum() / 3600))
print("info_duration hours all %.1f" % (idx.info_duration_s.sum() / 3600))
print("success values", idx.success.value_counts(dropna=False).to_dict())
d = use.groupby("day").duration_s.sum() / 3600
print("days", len(d), "first", d.index.min(), "last", d.index.max())
print("hours by month", use.assign(m=use.day.str[3:5]).groupby("m").duration_s.sum().div(3600).round(1).to_dict())
print("tac_level_l", idx.tac_level_l.value_counts().head(5).to_dict(), "tac_rate", idx.tac_rate_l_hz.value_counts().head(3).to_dict())
ep = root / use.iloc[0].rel_dir
print("==H5", ep)
with h5py.File(ep / "episode.h5", "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name} {obj.shape} {obj.dtype}")
        else:
            a = dict(obj.attrs)
            if a:
                print(f"  {name}/ attrs {list(a)[:8]}")
    f.visititems(show)
print("==TASK_INFO", (ep / "task_info.json").read_text()[:600])
