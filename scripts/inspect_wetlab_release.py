"""Compact read-only summary of a wetlab raw release (split schema, manifest task.name/trial.label,
admission index shape, README headline) -- used to onboard a new task without printing whole files."""
import json, sys, re, collections
from pathlib import Path

root = Path(sys.argv[1])
sp = json.load(open(root / "training_split.json"))
print("== training_split.json ==")
for k, v in sp.items():
    if isinstance(v, list):
        print(f"  {k}: list[{len(v)}] e.g. {v[0] if v else None}")
    elif isinstance(v, dict):
        print(f"  {k}: dict[{len(v)}] keys={list(v)[:6]}")
    else:
        print(f"  {k}: {str(v)[:300]}")
eps = sorted(p.name for p in root.iterdir() if re.fullmatch(r"[0-9a-f-]{36}", p.name))
print(f"== episode dirs: {len(eps)}")
names, labels, other = collections.Counter(), collections.Counter(), collections.Counter()
for e in eps:
    m = json.load(open(root / e / "robot" / "manifest.json"))
    names[m.get("task", {}).get("name")] += 1
    labels[str(m.get("trial", {}).get("label"))] += 1
print("  manifest task.name:", dict(names))
print("  manifest trial.label:", dict(labels))
adm = json.load(open(root / "admission_index.json"))
print("== admission_index.json:", type(adm).__name__, list(adm)[:6] if isinstance(adm, dict) else len(adm))
readme = (root / "README.md").read_text().splitlines()
print("== README (first 40 non-empty lines, truncated)")
for l in [l for l in readme if l.strip()][:40]:
    print("  " + l[:200])
sd = json.load(open(root / "safety_deployment.json"))
print("== safety_deployment.json top keys:", list(sd)[:12] if isinstance(sd, dict) else type(sd))
