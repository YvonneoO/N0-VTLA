#!/usr/bin/env python
"""Compact table of aggregate_stage1_scaling_posttrain_data_eval.py's summary.json.

Usage: print_stage1_scaling_posttrain_summary.py summary.json [metric ...]
Default metrics: i2t_top1. Prints mean [95% CI] and the paired delta vs the reference the
aggregation used (its --reference)."""
import json
import sys

d = json.load(open(sys.argv[1]))
metrics = sys.argv[2:] or ["i2t_top1"]
pct = {"i2t_top1", "i2t_top5", "i2t_top10", "i2t_top100", "t2i_top1", "t2i_top5", "t2i_top10", "t2i_top100"}
for metric in metrics:
    sc = 100 if metric in pct else 1
    print(f"\n##### {metric}" + (" (x100 = %)" if sc == 100 else ""))
    for split, labels in d.items():
        first = list(labels.values())[0]
        print(f"== {split} (pool {first['pool_size']}, {first['n_episodes']} eps, chance top1 {first['chance_top1'] * 100:.3f}%)")
        for label, e in labels.items():
            m = e["metrics"][metric]
            dd = e["delta_vs_reference"].get(metric)
            dtxt = f"d={dd['mean'] * sc:+.4f} [{dd['ci95'][0] * sc:+.4f},{dd['ci95'][1] * sc:+.4f}]" if dd else "(reference)"
            print(f"  {label:>7} {m['mean'] * sc:8.4f} [{m['ci95'][0] * sc:.4f},{m['ci95'][1] * sc:.4f}]  {dtxt}")
