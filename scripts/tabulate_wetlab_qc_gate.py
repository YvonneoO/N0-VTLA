"""Tabulate each episode's own qc_gate.json across the whole delivered batch.

Per ROBOT_POSTTRAIN_OPEN_ISSUES.md #5.1.7: don't trust the dataset card's
aggregate success-rate framing alone -- read what the vendor's own per-episode
QC gate actually recorded, and see which specific checks warn/fail and how often,
across every downloaded episode.

Usage:
  python scripts/tabulate_wetlab_qc_gate.py \
    /DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test \
    --output /DATA2/qianqian/n0vtla_robot_audit/qc_gate_tabulation_v1.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smoke_test_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = sorted(p for p in args.smoke_test_dir.iterdir() if p.is_dir() and p.name != "robot_model")
    rows = []
    verdict_counts: collections.Counter = collections.Counter()
    check_status_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    non_pass_examples: dict[str, list[dict]] = collections.defaultdict(list)

    for ep in episodes:
        qc_path = ep / "qc_gate.json"
        if not qc_path.exists():
            rows.append(dict(episode=ep.name, missing_qc_gate=True))
            continue
        qc = json.loads(qc_path.read_text())
        verdict_counts[qc.get("verdict")] += 1
        row = dict(episode=ep.name, verdict=qc.get("verdict"), n_fail=qc.get("n_fail"),
                   n_warn=qc.get("n_warn"), n_pass=qc.get("n_pass"), n_skip=qc.get("n_skip"))
        for check in qc.get("checks", []):
            name, status = check.get("name"), check.get("status")
            check_status_counts[name][status] += 1
            if status not in ("PASS",):
                non_pass_examples[name].append(dict(episode=ep.name, status=status, detail=check.get("detail")))
        rows.append(row)

    summary = dict(
        n_episodes=len(rows),
        verdict_counts=dict(verdict_counts),
        checks_with_any_non_pass={
            name: dict(status_counts=dict(counts),
                       examples=non_pass_examples[name][:5])
            for name, counts in check_status_counts.items()
            if set(counts) - {"PASS"}
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=2))

    print(f"episodes: {len(rows)}")
    print(f"verdicts: {dict(verdict_counts)}")
    print("checks with any non-PASS status across the batch:")
    for name, info in summary["checks_with_any_non_pass"].items():
        print(f"  {name}: {info['status_counts']}")
    print(f"full report written to {args.output}")


if __name__ == "__main__":
    main()
