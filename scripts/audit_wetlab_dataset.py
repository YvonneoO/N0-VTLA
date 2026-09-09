"""Multi-episode sanity sweep over a downloaded tacwam-wetlab-tasks smoke_test/ tree.

Checks two things across every episode, not just the one already hand-audited:
rig-side manifest consistency (ROBOT_POSTTRAIN_OPEN_ISSUES.md #1.1/#3.1), and
which of left_hand_data.npz / right_hand_data.npz actually carries live tactile
signal (#3.1) -- the swap confirmed on one episode is not assumed to generalize.

Usage:
  python scripts/audit_wetlab_dataset.py /DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test \
    --output /DATA2/qianqian/n0vtla_robot_audit/dataset_audit_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LIVE_STD_THRESHOLD = 1e-4


def pad_stats(npz_path: Path) -> dict:
    if not npz_path.exists():
        return dict(present=False, max_pad_std=None, n_pads=0)
    with np.load(npz_path, allow_pickle=False) as z:
        pad_keys = [k for k in z.files if k.startswith("tactile_")]
        stds = [float(np.std(np.asarray(z[k], np.float64))) for k in pad_keys]
    return dict(present=True, max_pad_std=max(stds) if stds else 0.0, n_pads=len(pad_keys))


def classify(left: dict, right: dict) -> str:
    # An absolute std threshold alone misclassifies same-block hard-linked episodes:
    # the dataset card documents that one capture is hard-linked into every trial of
    # a block, so the "dead" channel's own noise floor is a per-block constant, not a
    # per-episode one -- it can drift above a fixed absolute cutoff while still being
    # obviously the dead side of the pair. Compare the two channels to each other.
    l = left["max_pad_std"] or 0.0 if left["present"] else 0.0
    r = right["max_pad_std"] or 0.0 if right["present"] else 0.0
    if l < LIVE_STD_THRESHOLD and r < LIVE_STD_THRESHOLD:
        return "both_dead_or_missing"
    ratio = (l + 1e-12) / (r + 1e-12)
    if ratio >= 10:
        return "left_live_right_dead"
    if ratio <= 0.1:
        return "right_live_left_dead"
    return "both_live_ambiguous"


def audit_episode(ep_dir: Path) -> dict:
    row = dict(episode=ep_dir.name)
    manifest_path = ep_dir / "robot" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        sides = manifest.get("sides", {})
        row["task_name"] = manifest.get("task", {}).get("name")
        row["source_episode_id"] = manifest.get("source_episode_id")
        row["left_present"] = sides.get("left", {}).get("present")
        row["right_present"] = sides.get("right", {}).get("present")
        row["trial_label"] = manifest.get("trial", {}).get("label")
    else:
        row["manifest_missing"] = True

    task_info_path = ep_dir / "task_info.json"
    if task_info_path.exists():
        task_info = json.loads(task_info_path.read_text())
        row["task_info_name_en"] = task_info.get("name_en")
        row["task_info_success"] = task_info.get("success")

    left = pad_stats(ep_dir / "left_hand_data.npz")
    right = pad_stats(ep_dir / "right_hand_data.npz")
    row["left_npz"] = left
    row["right_npz"] = right
    row["tactile_classification"] = classify(left, right)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smoke_test_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = sorted(p for p in args.smoke_test_dir.iterdir() if p.is_dir() and p.name != "robot_model")
    rows = [audit_episode(ep) for ep in episodes]

    by_class: dict[str, int] = {}
    anomalies = []
    for row in rows:
        cls = row["tactile_classification"]
        by_class[cls] = by_class.get(cls, 0) + 1
        if row.get("left_present") is not False or row.get("right_present") is not True:
            anomalies.append(dict(episode=row["episode"], left_present=row.get("left_present"),
                                   right_present=row.get("right_present")))
        if row.get("task_name") != "cap_to_tray":
            anomalies.append(dict(episode=row["episode"], task_name=row.get("task_name")))
        if cls != "left_live_right_dead":
            anomalies.append(dict(episode=row["episode"], tactile_classification=cls,
                                   left_std=row["left_npz"]["max_pad_std"], right_std=row["right_npz"]["max_pad_std"]))

    summary = dict(
        n_episodes=len(rows),
        tactile_classification_counts=by_class,
        n_anomalies=len(anomalies),
        anomalies=anomalies,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=2))

    print(f"episodes audited: {len(rows)}")
    print(f"tactile classification counts: {by_class}")
    if anomalies:
        print(f"ANOMALIES ({len(anomalies)}) -- inspect before trusting a uniform left/right swap assumption:")
        for a in anomalies:
            print(" ", a)
    else:
        print("no anomalies: every episode is right-arm-only with live signal in left_hand_data.npz")
    print(f"full report written to {args.output}")


if __name__ == "__main__":
    main()
