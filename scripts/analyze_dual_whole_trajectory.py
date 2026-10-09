"""What a one-episode-per-trajectory build of the dual-arm release looks like (see dual_gap_bridge.fill_interior_gaps).
Usage: python scripts/analyze_dual_whole_trajectory.py RAW_DIR SPLIT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np

import dual_gap_bridge as gb
import n0_dual_adapter as nd

raw, split = Path(sys.argv[1]), Path(sys.argv[2])
uuids = json.loads(split.read_text())["train"]
tot = dict(traj=0, frames_now=0, filled=0, nonfinite=0, filled_neither=0, runs_after=0, frames=0)
gaps, lens, lost_edges = [], [], 0
for u in uuids:
    w = nd.resolve_dual_window(raw / u)
    now = sum(len(r) for r in w["runs"])
    v, added, nonfinite = gb.fill_interior_gaps(w)
    runs = nd.contiguous_runs(v)
    anyeng = w["engaged"]["right"] | w["engaged"]["left"]
    ids = np.flatnonzero(w["valid"])
    inner = np.zeros(len(v), bool); inner[ids[0]:ids[-1] + 1] = True
    filled = v & ~w["valid"]
    tot["traj"] += 1; tot["frames_now"] += now; tot["filled"] += added; tot["nonfinite"] += nonfinite
    tot["filled_neither"] += int((filled & ~anyeng).sum()); tot["runs_after"] += len(runs); tot["frames"] += sum(len(r) for r in runs)
    lens.append(sum(len(r) for r in runs))
    # longest contiguous filled stretch
    d = np.diff(np.concatenate([[0], filled.astype(int), [0]])); st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    gaps.append((en - st).max() if len(st) else 0)
print(json.dumps(tot, indent=1))
print("longest filled stretch per trajectory (rows): median", int(np.median(gaps)), "max", int(max(gaps)), "| rows>30 :", int(sum(g > 30 for g in gaps)), "trajectories")
print("episode (=trajectory) length frames min/med/max", int(min(lens)), int(np.median(lens)), int(max(lens)))
