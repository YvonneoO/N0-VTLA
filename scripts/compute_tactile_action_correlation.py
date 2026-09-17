"""3a — per-episode tactile-activity vs. predicted-action-quality correlation.

Paper-ready version of the artifact's §3a proxy method: pair each holdout
episode's tactile activity against that same episode's predicted-motion
quality (from the same causal open-loop inference eval_wetlab_ship_gate.py
uses), and test whether they correlate.

DESIGN NOTES (why this isn't the naive version):

- Tactile activity is computed from the ORIGINAL source `left_hand_data.npz`
  at the EXACT tactile-frame indices `wetlab_smoke_adapter.resolve_episode_window`
  selected when the canonical dataset was built -- not from the lossy encoded
  tactile video, and not from an independently re-derived window (which could
  silently drift from what the canonical dataset/eval actually consumed). Two
  canonical episodes can share one source uuid (a validity gap split it into
  two runs); each run is matched to its canonical episode by exact frame-count
  equality, asserted, not assumed by list order.
- Two independent tactile-activity metrics are computed (temporal variance,
  and contact-event rate against the SAME contact_threshold used to build the
  dataset) so the result isn't reported off a single, possibly cherry-picked,
  definition.
- Two prediction-quality channels are tested (arm xyz AND hand motor targets),
  since the causal ablation (artifact §2) found the real effect lives in the
  hand channel, not arm -- restricting to arm only (the artifact's original
  3a sketch) would likely find nothing and risk a false "no correlation"
  conclusion for the wrong reason.
- Inference is DENSE per episode (default stride=5, not eval_wetlab_ship_gate's
  default 60-samples-total-across-all-episodes cap) so each episode's own
  mean is a stable estimate, not 1-2 noisy samples.
- Both Pearson r and Spearman rho are reported. Significance uses an EXACT
  permutation test (n=7 holdout episodes -> all 7! = 5,040 permutations are
  enumerated, not approximated) -- the usual asymptotic-normal p-value is not
  valid at this sample size. A percentile bootstrap CI is also reported but
  is explicitly flagged as low-resolution at n=7 (few unique resamples).
- With 2 tactile metrics x 2 quality channels x 2 checkpoints = 8 tests,
  Holm-Bonferroni correction is applied across all 8 and reported alongside
  the raw per-test p-value -- a single small p out of 8 tests is not evidence
  by itself.

LIMITATIONS (report these alongside any result, do not drop them):
  - n=7 holdout episodes. Any correlation estimate here has wide uncertainty;
    this analysis is a hypothesis-generating probe, not a confirmatory test.
  - The quality proxy (mean predicted step size) is a SELF-CONSISTENCY check
    (is the motion demonstration-scale and non-oscillating), not error against
    the real future action -- see artifact §3a caveat and the module docstring
    of eval_wetlab_ship_gate.py. A true held-out-loss version (artifact §3b)
    is the confirmatory follow-up if this proxy looks interesting.
  - Some episodes share a source recording (one uuid split into two runs by a
    validity gap) or the same collection block -- they are not fully
    independent samples; this is reported in the output, not corrected for.

Usage:
  python scripts/compute_tactile_action_correlation.py \
    --config vtla_tactile_posttrain \
    --checkpoint checkpoints/vtla_tactile_posttrain/wetlab_v2_train_full_run1/20000 \
    --checkpoint-label official_base \
    --dataset-root /DATA2/qianqian/n0vtla_robot_audit/canonical_wetlab_v2_holdout \
    --raw-source-root /DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test \
    --norm-path /DATA2/qianqian/n0vtla_robot_audit/wetlab_tactile_norm_v2.json \
    --output /DATA2/qianqian/n0vtla_robot_audit/tactile_action_corr_official_base.json

  # quick pipeline-validation pass before the real run:
  python scripts/compute_tactile_action_correlation.py ... --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def step_stats(xyz: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Identical to eval_wetlab_ship_gate.py's step_stats -- duplicated here
    (not imported) so this script has no import-time dependency on argparse
    CLI setup in that module; kept byte-for-byte identical on purpose."""
    d = np.diff(xyz, axis=0)
    steps = np.linalg.norm(d, axis=1)
    u, v = d[:-1], d[1:]
    n = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    ok = n > 1e-6
    reversals = int(((u[ok] * v[ok]).sum(1) / n[ok] < 0).sum())
    return steps, reversals, int(ok.sum())


XYZ_SLICE = slice(10, 13)
HAND_SLICE = slice(20, 26)


def match_runs_to_episodes(runs: list[np.ndarray], episodes: list[dict]) -> dict[int, np.ndarray]:
    """episodes: canonical episodes sharing one source uuid, in ascending
    episode_index order. Matches each by EXACT frame-count equality against
    the source's own `runs` (from resolve_episode_window), not by list
    position -- fails loudly on any ambiguity instead of silently
    mispairing tactile activity with the wrong canonical episode."""
    lengths = [len(g) for g in runs]
    assigned: dict[int, np.ndarray] = {}
    remaining = list(range(len(runs)))
    for ep in episodes:
        candidates = [i for i in remaining if lengths[i] == ep["length"]]
        if len(candidates) != 1:
            raise ValueError(
                f"episode {ep['episode_index']} (uuid={ep['source_uuid']}, length={ep['length']}) "
                f"does not match exactly one run by length (candidates={candidates}, "
                f"run lengths={lengths}) -- refusing to guess the mapping"
            )
        assigned[ep["episode_index"]] = np.asarray(runs[candidates[0]])
        remaining.remove(candidates[0])
    if remaining:
        raise ValueError(f"{len(remaining)} source run(s) unmatched to any canonical episode: "
                          f"lengths={[lengths[i] for i in remaining]}")
    return assigned


def tactile_activity(taxel_grids: list[np.ndarray], baseline: np.ndarray, scale: np.ndarray,
                      threshold: np.ndarray) -> dict:
    """taxel_grids: one (n_frames, rows_p, cols_p) array per pad, in PAD_IDS
    order -- each pad is its own taxel GRID with a pad-specific grid shape
    (they are NOT uniform across pads, e.g. 16x10 vs 4x8, so this stays a
    list, never stacked into one array). baseline/scale/threshold are the
    per-pad scalar normalization stats, same order as taxel_grids.

    Two independent, length-normalized (mean/rate, not sum) metrics:
      - variance: temporal variance of each taxel cell across the episode's
        frames, averaged over every cell of every pad.
      - contact_rate: fraction of frames where ANY cell of ANY pad crosses
        that pad's own contact_threshold (same threshold the dataset build
        used, applied uniformly across a pad's grid -- the normalization
        has no per-cell resolution, only per-pad)."""
    n_frames = taxel_grids[0].shape[0]
    pad_variances = []
    in_contact_any_pad = np.zeros(n_frames, dtype=bool)
    for grid, b, s, t in zip(taxel_grids, baseline, scale, threshold):
        pad_variances.append(float(np.mean(np.var(grid, axis=0))))
        normalized = (grid - b) / s
        in_contact_any_pad |= np.any(normalized > t, axis=(1, 2))
    variance = float(np.mean(pad_variances))
    contact_rate = float(np.mean(in_contact_any_pad))
    return dict(variance=variance, contact_rate=contact_rate, n_frames=int(n_frames))


def exact_permutation_p(x: np.ndarray, y: np.ndarray, statistic) -> float:
    """Exact two-sided permutation test: enumerate every permutation of y's
    order (n! for n<=9; falls back to 200,000 random permutations above that,
    which does not apply at n=7 here) and compare |statistic| against the
    observed value. Valid at small n where asymptotic p-values are not."""
    n = len(x)
    observed = abs(statistic(x, y))
    if n > 9:
        rng = np.random.default_rng(0)
        count = 0
        total = 200_000
        for _ in range(total):
            perm = rng.permutation(y)
            if abs(statistic(x, perm)) >= observed:
                count += 1
        return count / total
    count = 0
    total = 0
    for perm in itertools.permutations(y):
        total += 1
        if abs(statistic(x, np.asarray(perm))) >= observed:
            count += 1
    return count / total


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, statistic, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    n = len(x)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = statistic(x[idx], y[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down. Returns adjusted p-values in the ORIGINAL
    input order (not sorted), each already monotone-enforced."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="vtla_tactile_posttrain")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True, help="e.g. official_base, ours")
    parser.add_argument("--dataset-root", type=Path, required=True, help="canonical holdout dataset")
    parser.add_argument("--raw-source-root", type=Path, required=True, help="raw per-uuid episode dirs")
    parser.add_argument("--norm-path", type=Path, required=True, help="tactile normalization used to build the dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="cap_to_tray")
    parser.add_argument("--stride", type=int, default=5, help="frames between sampled inference windows within an episode")
    parser.add_argument("--margin", type=int, default=10, help="frames to skip at each episode's start/end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                         help="tiny pipeline-validation pass: first 2 episodes only, stride=50")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch

    import wetlab_smoke_adapter as adapter
    from n0vtla.policies import policy_config
    from n0vtla.training import config as _config
    from n0vtla.training.data_loader import LocalLeRobotV3Dataset

    norm = json.loads(args.norm_path.read_text())
    baseline = np.asarray(norm["normal_baseline"][15:30], dtype=np.float64)
    scale = np.asarray(norm["normal_scale"][15:30], dtype=np.float64)
    threshold = np.asarray(norm["contact_threshold"][15:30], dtype=np.float64)
    from itw_pressure import PAD_IDS

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    if args.smoke:
        episodes = episodes[:2]
    offsets = np.concatenate([[0], np.cumsum([json.loads(l)["length"]
                              for l in (args.dataset_root / "meta" / "episodes.jsonl").open()])[:-1]])

    by_uuid: dict[str, list[dict]] = {}
    for ep in episodes:
        by_uuid.setdefault(ep["source_uuid"], []).append(ep)
    for uuid_episodes in by_uuid.values():
        uuid_episodes.sort(key=lambda e: e["episode_index"])

    print(f"=== resolving tactile activity for {len(episodes)} episode(s) "
          f"across {len(by_uuid)} source recording(s) ===")
    tactile_by_episode: dict[int, dict] = {}
    shared_uuid_note: dict[int, str] = {}
    for uuid, uuid_episodes in by_uuid.items():
        source = args.raw_source_root / uuid
        window = adapter.resolve_episode_window(source, task_name=args.task_name, require_success_label=True)
        run_map = match_runs_to_episodes(window["runs"], uuid_episodes)
        with np.load(window["tactile_source"], allow_pickle=False) as z:
            for ep in uuid_episodes:
                g = run_map[ep["episode_index"]]
                ti = window["ti"][g]
                taxel_grids = [np.asarray(z[f"tactile_{p}"], np.float64)[ti] for p in PAD_IDS]
                tactile_by_episode[ep["episode_index"]] = tactile_activity(taxel_grids, baseline, scale, threshold)
                if len(uuid_episodes) > 1:
                    shared_uuid_note[ep["episode_index"]] = uuid
        print(f"  uuid={uuid}: {len(uuid_episodes)} episode(s) matched by exact run length")

    print(f"=== loading policy from {args.checkpoint} ===")
    policy = policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint,
                                                  pytorch_device=args.device)
    dataset = LocalLeRobotV3Dataset(args.dataset_root)

    stride = 50 if args.smoke else args.stride
    print(f"=== running dense causal inference (stride={stride}, margin={args.margin}) ===")
    quality_by_episode: dict[int, dict] = {}
    for ep in episodes:
        idx = ep["episode_index"]
        length = ep["length"]
        frames = list(range(args.margin, max(args.margin + 1, length - args.margin), stride))
        arm_means, hand_means = [], []
        for frame_index in frames:
            global_index = int(offsets[idx]) + frame_index
            item = dataset[global_index]
            item.pop("action", None)
            out = policy.infer(item)
            actions = np.asarray(out["actions"])
            arm_steps, _, arm_n = step_stats(actions[:, XYZ_SLICE])
            hand_steps, _, hand_n = step_stats(actions[:, HAND_SLICE])
            if arm_n > 0:
                arm_means.append(float(arm_steps.mean()))
            if hand_n > 0:
                hand_means.append(float(hand_steps.mean()))
        quality_by_episode[idx] = dict(
            arm_mean_step=float(np.mean(arm_means)) if arm_means else float("nan"),
            hand_mean_step=float(np.mean(hand_means)) if hand_means else float("nan"),
            n_samples=len(frames),
        )
        print(f"  episode {idx}: {len(frames)} samples, "
              f"arm_mean_step={quality_by_episode[idx]['arm_mean_step']:.3f}, "
              f"hand_mean_step={quality_by_episode[idx]['hand_mean_step']:.3f}")

    per_episode = []
    for ep in episodes:
        idx = ep["episode_index"]
        per_episode.append(dict(
            episode_index=idx, source_uuid=ep["source_uuid"], block=ep.get("block"),
            length=ep["length"], shares_source_recording_with=shared_uuid_note.get(idx),
            **tactile_by_episode[idx], **quality_by_episode[idx],
        ))

    tactile_metrics = ["variance", "contact_rate"]
    quality_channels = ["arm_mean_step", "hand_mean_step"]
    results = []
    raw_pvalues = []
    for tm in tactile_metrics:
        for qc in quality_channels:
            x = np.array([row[tm] for row in per_episode])
            y = np.array([row[qc] for row in per_episode])
            r = pearson(x, y)
            rho = spearman(x, y)
            p_pearson = exact_permutation_p(x, y, pearson) if not args.smoke else float("nan")
            p_spearman = exact_permutation_p(x, y, spearman) if not args.smoke else float("nan")
            ci = bootstrap_ci(x, y, pearson, seed=args.seed) if not args.smoke else (float("nan"), float("nan"))
            results.append(dict(tactile_metric=tm, quality_channel=qc, n=len(x),
                                 pearson_r=r, pearson_p_exact=p_pearson, pearson_ci95=ci,
                                 spearman_rho=rho, spearman_p_exact=p_spearman))
            raw_pvalues.append(p_pearson if not np.isnan(p_pearson) else 1.0)

    if not args.smoke:
        holm = holm_correction(raw_pvalues)
        for res, adj in zip(results, holm):
            res["pearson_p_holm_corrected_across_all_tests_this_checkpoint"] = adj

    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_label=args.checkpoint_label,
        dataset_root=str(args.dataset_root), norm_path=str(args.norm_path),
        smoke=args.smoke, stride=stride, margin=args.margin, n_episodes=len(episodes),
        per_episode=per_episode, correlations=results,
        limitations=[
            f"n={len(episodes)} holdout episodes -- hypothesis-generating, not confirmatory.",
            "Quality proxy is predicted-motion self-consistency (demonstration-scale, non-oscillating), "
            "not error against the real future action -- see eval_wetlab_ship_gate.py's own docstring.",
            "Some episodes share a source recording or collection block (see per_episode."
            "shares_source_recording_with / block) -- not fully independent samples.",
            "4 tests per checkpoint (2 tactile metrics x 2 quality channels); Holm-corrected p-values "
            "are the ones to trust, not the raw per-test p-value.",
            "Bootstrap CI is low-resolution at n=7 (few unique resamples) -- read as a rough band, not a precise interval.",
        ] if not args.smoke else ["SMOKE RUN -- 2 episodes only, stats are not meaningful, pipeline validation only."],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(checkpoint_label=args.checkpoint_label, correlations=results), indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
