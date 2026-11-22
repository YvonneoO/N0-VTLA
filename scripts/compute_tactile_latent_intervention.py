"""4 -- causal intervention on the tactile latent z, plus a per-STEP refinement of 3c.

Two results from one pass over each checkpoint's held-out split:

(A) PER-STEP correlation: tactile-predictor InfoNCE quality (as in 3c) vs. real held-out
    action loss (as in 3b), paired at the SAMPLE level instead of averaged down to 7
    per-episode means first. n is the pool size (hundreds), not 7 -- a much better-powered
    version of exactly the same question 3c asked at episode granularity.

(B) LATENT INTERVENTION: on the SAME held-out samples (same vision/language/state/ground-truth
    action), only the tactile latent fed into the action expert is changed, and the resulting
    real flow-matching action loss is compared:
      normal   z = z_hat        (the model's own real prediction)          -> L_normal
      zero     z = 0            (bypass the tactile pathway's output)      -> L_zero
      shuffle  z = z_hat(j)     (a real z from a DIFFERENT sample, j != i) -> L_shuffle
      oracle   z = z*_future    (the real future-tactile target, pooled+broadcast; an upper
                                  bound only -- future information leakage, never a deployment
                                  setting)                                 -> L_oracle
    This is a genuinely causal test of "tactile latent -> better action generation", the one
    link 3b (tactile activity -> action-critical episodes) and 3c (Stage-1 pretraining -> better
    tactile forecasting) don't establish on their own: does the ACTION EXPERT actually do
    something useful with a better z, on the SAME inputs otherwise?

WHERE THE INTERVENTION HAPPENS: the model's real forward() computes
  z, g, has_tac = self._compute_z(vl_ctx, prefix_pad_masks)
  z_w = self.z_proj(z); suffix = self._embed_suffix_with_z(state, x_t, time, z_w, g=g, ...)
and _embed_suffix_with_z applies z_gate (z = z * gate) before injecting z_w as suffix tokens.
This script monkeypatches ONLY `model._compute_z` for the duration of one forward() call --
it still runs the real, unmodified `_compute_z` internally to get the real (g, has_tac), and
substitutes only the returned `z` tensor -- so g/has_tac/z_gate/z_proj/everything downstream
runs exactly as it would for a real rollout, on whichever z tensor was substituted in.

WHY z_gate MUST be forced on for THIS script (unlike 3c): z_gate is only instantiated by
N0VTLAPolicy.__init__ when `config.z_gate_zero_init=True` -- a config flag, NOT implied by
`stage1_pretrain_enabled`. The Stage-1 predictor-pretrain TrainConfig used here (to get a model
skeleton whose forward() can also build the future-tactile target) does not set this flag, so a
plain Stage-1 skeleton has NO z_gate parameter at all (confirmed by 3c's harmless `unexpected=
['z_gate']` when loading a real checkpoint into it). 3c never reached `_embed_suffix_with_z`, so
that gap was harmless there. This script DOES run the real action-injection path, so it forces
`z_gate_zero_init=True` on the model_cfg before construction -- this only changes whether the
`z_gate` PARAMETER exists in the skeleton (so the checkpoint's real trained value loads into it
correctly, with zero missing/unexpected keys related to z_gate); it changes nothing about which
checkpoint weights get loaded or how forward() computes.

WHY A SEPARATE (zero, shuffle, oracle) SET OF CONDITIONS, NOT JUST 2:
  - zero vs. normal is the causal analogue of section 2's raw-tactile-zeroing ablation, but
    cleaner: it zeros the LATENT at the exact point the action expert consumes it, rather than
    zeroing the raw tactile encoder input (which may not produce an exactly-zero z downstream).
  - shuffle vs. normal is the sharper test: it keeps z's scale/distribution identical (a real
    predicted z, just from the wrong sample) and asks whether the action expert needs a z that
    matches the CURRENT physical context, or merely needs "some tactile-shaped vector present".
    zero alone cannot distinguish "tactile content matters" from "z is just an out-of-distribution
    input the expert reacts to".
  - oracle vs. normal is a diagnostic ceiling, not a deployment setting (it leaks the real future):
    if L_oracle < L_normal < L_shuffle/zero, the action expert can exploit a MORE ACCURATE tactile
    forecast than it currently gets -- i.e. improving the predictor's forecast quality (3c's axis)
    would plausibly improve action prediction, closing the loop back to 3c's null correlation.

COMMON RANDOM NUMBERS (CRN): for a given sample and repeat index, the SAME (noise, time) draw is
reused across all four conditions (draw once, pass explicitly via `model(obs, actions, noise=,
time=)`), so a paired comparison isolates the effect of the z substitution alone rather than
extra between-condition draw variance. This makes the paired significance tests below (Monte
Carlo sign-flip on the per-sample differences) far more powerful than an unpaired comparison.

STATISTICS:
  - (A) per-step correlation: Pearson r / Spearman rho with a Monte Carlo permutation p-value
    (the exact enumeration used elsewhere in this project is only tractable up to n~10; at
    n=hundreds this script permutes the pairing `--n-resamples` times instead) plus a bootstrap CI
    (reusing the same `bootstrap_ci` helper 3a/3b/3c use).
  - (B) intervention: paired Monte Carlo sign-flip test per condition-vs-normal comparison
    (valid at any n, exact under the paired-difference null), Holm-corrected across the 3
    comparisons (zero, shuffle, oracle, each vs. normal) per checkpoint.

LIMITATIONS:
  - Oracle substitution broadcasts the mean-pooled z* to all n_latent token slots, discarding
    whatever per-token structure z normally has -- a documented simplification, not the model's
    native latent structure.
  - "Shuffle" draws a fixed, seeded derangement over the SAME checkpoint's own pool -- the
    specific mismatched z each sample receives is arbitrary (one specific wrong-sample pairing,
    not an average over all possible mismatches).
  - Still n=1 checkpoint-run per condition (no retraining) -- this tests what a FROZEN action
    expert does with a substituted latent, not whether training WITH better latents would help
    more.

Usage:
  python scripts/compute_tactile_latent_intervention.py \
    --checkpoint checkpoints/vtla_tactile_posttrain/wetlab_v2_train_full_run1/20000 \
    --checkpoint-label official_base \
    --dataset-root /DATA2/qianqian/n0vtla_robot_audit/canonical_wetlab_v2_holdout \
    --asset-id wetlab_v2_train \
    --output /DATA2/qianqian/n0vtla_robot_audit/tactile_intervention_official_base.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_tactile_action_correlation import bootstrap_ci, holm_correction, pearson, spearman


def monte_carlo_permutation_p(x: np.ndarray, y: np.ndarray, stat_fn, n_resamples: int, seed: int) -> float:
    """Two-sided permutation p-value for a large n where exact enumeration is intractable.
    Same logic as the exact test used elsewhere (permute the pairing, compare |stat|), just
    Monte Carlo instead of full enumeration."""
    rng = np.random.default_rng(seed)
    observed = abs(stat_fn(x, y))
    y = np.asarray(y)
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(len(y))
        if abs(stat_fn(x, y[perm])) >= observed:
            count += 1
    return (count + 1) / (n_resamples + 1)  # +1/+1: never report exactly p=0


def paired_sign_flip_p(diffs: np.ndarray, n_resamples: int, seed: int) -> float:
    """Two-sided Monte Carlo sign-flip test on paired differences (valid at any n; exact under
    the paired-difference null of a symmetric-about-zero distribution)."""
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(diffs)))
    n = len(diffs)
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(float(np.mean(diffs * signs))) >= observed:
            count += 1
    return (count + 1) / (n_resamples + 1)


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError(f"need at least 2 pooled samples for a derangement, got {n}")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


@contextlib.contextmanager
def override_z(model, mode: str, z_source=None):
    """Monkeypatch model._compute_z for one forward() call: the real (g, has_tac) always come
    from the real, unmodified computation; only the returned z is substituted. 'normal' returns
    the real z unchanged (kept as an explicit branch for symmetry with the other 3 conditions,
    not a shortcut -- byte-identical to not patching at all)."""
    original = model._compute_z

    def patched(vl_ctx, prefix_pad_masks=None):
        real_z, g, has_tac = original(vl_ctx, prefix_pad_masks)
        if mode == "normal":
            z_out = real_z
        elif mode == "zero":
            z_out = torch.zeros_like(real_z)
        elif mode in ("shuffle", "oracle"):
            n_latent = real_z.shape[1]
            src = z_source.to(device=real_z.device, dtype=real_z.dtype)
            if src.dim() == 1:  # oracle: (D,) pooled vector -> broadcast to (1, n_latent, D)
                z_out = src.view(1, 1, -1).expand(1, n_latent, -1).contiguous()
            else:  # shuffle: (n_latent, D) real z from another sample -> (1, n_latent, D)
                z_out = src.unsqueeze(0)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return z_out, g, has_tac

    model._compute_z = patched
    try:
        yield
    finally:
        model._compute_z = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-config", default="vtla_stage1_predictor_pretrain")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4, help="common-random-number (noise,time) draws averaged per sample")
    parser.add_argument("--n-resamples", type=int, default=20000, help="Monte Carlo resamples for permutation/sign-flip tests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import os
    for var in ("VTLA_ASSET_ID", "VTLA_DATASET_PATH"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} must be exported in the shell before running this script "
                                f"(config-construction reads it at import time)")

    import torch
    import torch.nn.functional as F
    import safetensors.torch
    import jax
    import scripts.train_n0vtla  # noqa: F401 -- patches PI0Pytorch -> N0VTLAPolicy
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = [ep["length"] for ep in episodes]
    total_rows = sum(lengths)
    boundaries = np.concatenate([[0], np.cumsum(lengths)])

    def episode_of(running_index):
        if running_index >= total_rows:
            return None
        return int(np.searchsorted(boundaries, running_index, side="right") - 1)

    device = torch.device(args.device)
    config = _config.get_config(args.stage1_config)
    object.__setattr__(config, "batch_size", 1)
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    # Force z_gate into existence so the checkpoint's real trained gate loads and the real
    # gated injection path runs -- see module docstring.
    object.__setattr__(model_cfg, "z_gate_zero_init", True)

    print(f"=== loading checkpoint ({args.checkpoint_label}) from {args.checkpoint} ===")
    import n0vtla.models_pytorch.pi0_pytorch
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(
        model, str(args.checkpoint / "model.safetensors"), device=str(device)
    )
    # Only tactile_recon_head may be missing (never trained under post-train; this script never
    # calls it) -- z_gate is no longer expected to be unexpected now that z_gate_zero_init=True.
    missing_ok = [k for k in missing if "tactile_recon_head" not in k]
    if missing_ok or unexpected:
        raise RuntimeError(f"checkpoint load mismatch beyond the expected tactile_recon_head gap -- "
                            f"missing={missing_ok}, unexpected={unexpected}")
    if missing:
        print(f"  (expected) tactile_recon_head not in checkpoint, {len(missing)} key(s) "
              f"randomly initialized and unused -- this script never calls the reconstruction head")
    print("  z_gate loaded with 0 missing/unexpected keys -- real gated injection path is active")
    model.eval()

    print("=== building held-out data loader (stage1 DataConfig, shuffle=False, batch_size=1) ===")
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    underlying = loader._data_loader.torch_loader.dataset
    counted = len(underlying)
    print(f"  dataset reports {counted} samples; episodes.jsonl sums to {total_rows}")
    if counted != total_rows:
        raise RuntimeError(f"dataset length {counted} != episodes.jsonl total {total_rows} -- aborting")

    stride = 50 if args.smoke else args.stride
    repeats = 1 if args.smoke else args.repeats
    n_resamples = 500 if args.smoke else args.n_resamples

    # ---------------- PASS 1: collect (z, z*) for the whole held-out split ----------------
    print(f"=== pass 1/2: collecting (z, z*) embeddings (stride={stride}) ===")
    z_raw_pool, zstar_raw_pool = [], []          # for shuffle / oracle substitution sources
    z_norm_pool, zstar_norm_pool = [], []        # for the InfoNCE quality score
    ep_idx_pool, valid_running_index = [], {}    # running_index -> pool position
    running_index = 0
    for observation, _actions in loader:
        ep_idx = episode_of(running_index)
        if ep_idx is None:
            break
        if running_index % stride == 0:
            obs = jax.tree.map(lambda x: x.to(device), observation)
            with torch.no_grad():
                images, img_masks, lang_tokens, lang_masks, _s, _e = model._preprocess_observation(obs, train=True)
                vl_ctx, _pe, prefix_pad_masks, _pam, _pkv = model._prefix_forward(
                    images, img_masks, lang_tokens, lang_masks, use_cache=False
                )
                z, _g, has_tac = model._compute_z(vl_ctx, prefix_pad_masks)
                tac_f = model._last_tac_f or {}
                tac_t = model._last_tac_t or {}
                tac_mask_f = getattr(model, "_last_tac_mask_f", None)
                z_star, _dbar, has_future = model._build_future_target(tac_f, tac_t, tac_mask_f)
            if z_star is not None and bool((has_tac & has_future).all()):
                pool_i = len(z_raw_pool)
                z_raw_pool.append(z[0].detach().float().cpu())                                   # (n_latent, D)
                zs_raw = z_star.float().mean(dim=1)[0].detach().cpu()                              # (D,)
                zstar_raw_pool.append(zs_raw)
                z_norm_pool.append(F.normalize(z[0].float().mean(dim=0, keepdim=True), dim=-1)[0].cpu())  # (D,)
                zstar_norm_pool.append(F.normalize(zs_raw.unsqueeze(0), dim=-1)[0].cpu())
                ep_idx_pool.append(ep_idx)
                valid_running_index[running_index] = pool_i
        running_index += 1
        if running_index >= total_rows:
            break

    n_pool = len(z_raw_pool)
    print(f"  collected {n_pool} valid samples")
    if n_pool < 2:
        raise RuntimeError(f"only {n_pool} valid sample(s) -- cannot build a pool")

    temp = float(getattr(model_cfg, "stage1_temperature", 1.0))
    Z = torch.stack(z_norm_pool)          # (N, D)
    Zs = torch.stack(zstar_norm_pool)     # (N, D)
    logits = (Z @ Zs.t()) / temp
    labels = torch.arange(n_pool)
    loss_i2t = F.cross_entropy(logits, labels, reduction="none")
    loss_t2i = F.cross_entropy(logits.t(), labels, reduction="none")
    infonce_per_sample = (0.5 * (loss_i2t + loss_t2i)).numpy()

    rng = np.random.default_rng(args.seed)
    derangement = random_derangement(n_pool, rng)

    # ---------------- PASS 2: 4-condition action-loss forward per sample ----------------
    print(f"=== pass 2/2: intervention forward passes (repeats={repeats}, conditions=4) ===")
    conditions = ["normal", "zero", "shuffle", "oracle"]
    per_sample_loss = {c: np.full(n_pool, np.nan) for c in conditions}
    running_index = 0
    for observation, actions in loader:
        ep_idx = episode_of(running_index)
        if ep_idx is None:
            break
        pool_i = valid_running_index.get(running_index)
        if pool_i is not None:
            obs = jax.tree.map(lambda x: x.to(device), observation)
            actions_t = actions.to(torch.float32).to(device)
            draws = {c: [] for c in conditions}
            for r in range(repeats):
                torch.manual_seed(args.seed * 100_000 + running_index * 10 + r)
                noise_r = model.sample_noise(actions_t.shape, device)
                time_r = model.sample_time(actions_t.shape[0], device)
                for c in conditions:
                    if c == "shuffle":
                        src = z_raw_pool[derangement[pool_i]]
                    elif c == "oracle":
                        src = zstar_raw_pool[pool_i]
                    else:
                        src = None
                    with override_z(model, c, src), torch.no_grad():
                        losses = model(obs, actions_t, noise=noise_r, time=time_r)
                    per_sample_scalar = losses.reshape(losses.shape[0], -1).mean(dim=1)
                    draws[c].append(float(per_sample_scalar.item()))
            for c in conditions:
                per_sample_loss[c][pool_i] = float(np.mean(draws[c]))
            if pool_i % 50 == 0:
                print(f"  scored {pool_i + 1}/{n_pool} samples "
                      f"(normal={per_sample_loss['normal'][pool_i]:.4f}, "
                      f"zero={per_sample_loss['zero'][pool_i]:.4f}, "
                      f"shuffle={per_sample_loss['shuffle'][pool_i]:.4f}, "
                      f"oracle={per_sample_loss['oracle'][pool_i]:.4f})")
        running_index += 1
        if running_index >= total_rows:
            break

    # ---------------- (A) per-step correlation: InfoNCE quality vs. real action loss ----------------
    x = infonce_per_sample
    y = per_sample_loss["normal"]
    step_corr = dict(
        n=n_pool, pearson_r=pearson(x, y), spearman_rho=spearman(x, y),
        pearson_p_mc=monte_carlo_permutation_p(x, y, pearson, n_resamples, args.seed),
        spearman_p_mc=monte_carlo_permutation_p(x, y, spearman, n_resamples, args.seed + 1),
        pearson_ci95=bootstrap_ci(x, y, pearson, seed=args.seed),
    )

    # ---------------- (B) intervention table ----------------
    intervention = {}
    raw_p = []
    comparisons = ["zero", "shuffle", "oracle"]
    for c in comparisons:
        diffs = per_sample_loss[c] - per_sample_loss["normal"]
        p = paired_sign_flip_p(diffs, n_resamples, args.seed)
        raw_p.append(p)
        intervention[c] = dict(
            mean_loss=float(np.mean(per_sample_loss[c])),
            mean_diff_vs_normal=float(np.mean(diffs)),
            p_sign_flip=p,
        )
    holm = holm_correction(raw_p)
    for c, adj in zip(comparisons, holm):
        intervention[c]["p_holm_corrected"] = adj
    intervention["normal"] = dict(mean_loss=float(np.mean(per_sample_loss["normal"])))

    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_label=args.checkpoint_label,
        dataset_root=str(args.dataset_root), smoke=args.smoke, stride=stride, repeats=repeats,
        n_pool=n_pool, n_resamples=n_resamples,
        per_step_correlation=step_corr,
        intervention=intervention,
        limitations=[
            "Oracle substitution broadcasts the mean-pooled z* to all n_latent token slots -- "
            "a documented simplification, not the model's native per-token latent structure.",
            "Shuffle uses one fixed, seeded derangement over this checkpoint's own pool -- one "
            "specific mismatched pairing per sample, not an average over all possible mismatches.",
            "Tests a FROZEN action expert's response to a substituted latent, not whether "
            "training with better latents would help more.",
            f"Monte Carlo tests ({n_resamples} resamples) approximate exact permutation/sign-flip "
            "p-values -- exact enumeration is intractable at this n.",
        ] if not args.smoke else ["SMOKE RUN -- pipeline validation only, stats not meaningful."],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(checkpoint_label=args.checkpoint_label, per_step_correlation=step_corr,
                           intervention=intervention), indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
