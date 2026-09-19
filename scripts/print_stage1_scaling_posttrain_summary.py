#!/usr/bin/env python
"""Compact table of aggregate_stage1_scaling_posttrain_data_eval.py's summary.json."""
import json
import sys

d = json.load(open(sys.argv[1]))
for split, labels in d.items():
    print(f"\n== {split} ==")
    print(f"{'ckpt':>7} {'pool':>5} {'eps':>4} | i2t top1 [95%CI] | d vs base [95%CI] | i2t top10 | i2t MRR | t2i top1 | t2i top10 | pos_cos | pool_nce")
    for label, e in labels.items():
        m, dl = e["metrics"], e["delta_vs_reference"]
        t1 = m["i2t_top1"]
        dd = dl.get("i2t_top1")
        dtxt = f"{dd['mean'] * 100:+.2f} [{dd['ci95'][0] * 100:+.2f},{dd['ci95'][1] * 100:+.2f}]" if dd else "-"
        print(f"{label:>7} {e['pool_size']:>5} {e['n_episodes']:>4} | {t1['mean'] * 100:5.2f} [{t1['ci95'][0] * 100:.2f},{t1['ci95'][1] * 100:.2f}] | {dtxt} | "
              f"{m['i2t_top10']['mean'] * 100:5.2f} | {m['i2t_mrr']['mean']:.4f} | {m['t2i_top1']['mean'] * 100:5.2f} | "
              f"{m['t2i_top10']['mean'] * 100:5.2f} | {m['pos_cos']['mean']:.3f} | {m['pool_nce']['mean']:.3f}")
    print(f"  chance top1 = {list(labels.values())[0]['chance_top1'] * 100:.3f}%")
