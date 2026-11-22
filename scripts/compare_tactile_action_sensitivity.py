#!/usr/bin/env python
"""How much does the tactile input move the sampled actions?  Compares three dumps (dump_posttrain_sampled_actions.py) of the same checkpoint,
rows and noise, fed with real / predicted / all-zero tactile. Physical units: xyz = L2 over the 3 position dims (mm), rot6d = mean abs error over its
dims, hand = mean abs over the 6 hand dims (motor counts 0..1000). Reported over all scored rows and over CONTACT rows (the real glove input
differs from its first-frame baseline by at least --act-thr in [-1,1] image units, 0.067 = x of 0.3 in policy units), as the mean over episodes:
  |real - pred|, |real - zero|, |pred - zero|     difference between the sampled chunks (the sensitivity of the actions to the tactile input)
  err(real) err(pred) err(zero)                  error of each variant against the demonstrated chunk
  python compare_tactile_action_sensitivity.py --real a.npz --pred b.npz --zero c.npz [--out report.md]
"""
import argparse
import json

import numpy as np

G = {"xyz": slice(10, 13), "rot6d": slice(13, 19), "hand": slice(20, 26)}


def dist(a, b):
    e = a - b                                                   # (n, H, D)
    return {"xyz_mm": np.linalg.norm(e[:, :, G["xyz"]], axis=2).mean(1), "rot6d": np.abs(e[:, :, G["rot6d"]]).mean((1, 2)),
            "hand": np.abs(e[:, :, G["hand"]]).mean((1, 2))}


def macro(v, ep, sel):
    es = [v[sel & (ep == e)].mean() for e in np.unique(ep) if (sel & (ep == e)).any()]
    return (float(np.mean(es)), float(np.std(es, ddof=1) / np.sqrt(len(es))) if len(es) > 1 else float("nan"), len(es))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--zero", required=True)
    ap.add_argument("--act-thr", type=float, default=0.067)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R, P, Z = (np.load(f) for f in (a.real, a.pred, a.zero))
    assert (R["rows"] == P["rows"]).all() and (R["rows"] == Z["rows"]).all(), "dumps cover different rows"
    ep, act = R["episode"], R["tactile_act"]
    print(f"{a.label}: {len(ep)} rows, {len(np.unique(ep))} episodes; contact rows (real tactile act >= {a.act_thr}): {int((act >= a.act_thr).sum())}")
    print(f"tactile activity per row of the real input: median {np.median(act):.3f}, p90 {np.percentile(act, 90):.3f}, max {act.max():.3f}; "
          f"pred input: median {np.median(P['tactile_act']):.3f}, zero input: max {Z['tactile_act'].max():.3f}")
    sets = {"all rows": np.ones(len(ep), bool), "contact rows": act >= a.act_thr, "no-contact rows": act < a.act_thr}
    pairs = {"|real-pred|": dist(R["pred"], P["pred"]), "|real-zero|": dist(R["pred"], Z["pred"]), "|pred-zero|": dist(P["pred"], Z["pred"]),
             "err real": dist(R["pred"], R["gt"]), "err pred": dist(P["pred"], R["gt"]), "err zero": dist(Z["pred"], R["gt"])}
    lines = [f"| rows | n rows | quantity | xyz (mm) | rot6d | hand (counts) |", "|---|---|---|---|---|---|"]
    out = {}
    for sn, sel in sets.items():
        if not sel.any():
            continue
        for pn, d in pairs.items():
            m = {k: macro(v, ep, sel) for k, v in d.items()}
            out[f"{sn}/{pn}"] = {k: m[k][0] for k in m}
            lines.append(f"| {sn} | {int(sel.sum())} | {pn} | " + " | ".join(f"{m[k][0]:.3f} ± {m[k][1]:.3f}" for k in ("xyz_mm", "rot6d", "hand")) + " |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(f"# {a.label}\n\n" + txt + "\n")
        open(a.out.replace(".md", ".json"), "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
