#!/usr/bin/env python
"""Held-out Stage-1 (online) eval for the data-scaling study, one checkpoint at a time.

WHY THIS EXISTS (not just reading each run's own training loss off wandb): the
20/40/60/80pct scaling runs all train for the SAME number of steps (14000) on
DIFFERENT-sized episode pools drawn from ONE shuffle (see build_itw_scaling_splits.py).
A smaller pool gets recycled far more times over those 14000 steps than a larger one,
so a smaller-fraction run's own training loss is biased low by repetition/memorization
of its own pool -- it is NOT a fair proxy for "how well did this checkpoint learn to
predict future tactile state in general". The only fair comparison is every
checkpoint's loss on the SAME episodes none of them ever trained on: the "held_out"
key build_itw_scaling_splits.py writes (the complement of the largest built fraction,
so it is by construction absent from every 20/40/60/80pct split).

This script scores exactly one checkpoint per invocation (run it once per data
fraction, pointing --checkpoint-dir at that fraction's exp dir -- the LATEST step
found there is loaded, matching load_stage1_checkpoint's own resume behavior) and
writes one small JSON report. Collect the reports across fractions and plot
data_fraction (x, log scale) vs. mean_stage1_nce or mean_stage1_total (y) for the
scaling-law curve.

Reuses train_stage1_online.py's own model-construction, checkpoint-loading, and loss
computation UNCHANGED (imported, not copied) so the held-out number is computed by
the exact same code path as training, just in eval mode with one full held-out pass
instead of a shuffled infinite training stream.

Usage:
    python scripts/compute_stage1_scaling_held_out_eval.py \
        --checkpoint-dir /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract/checkpoints/vtla_stage1_predictor_pretrain/stage1_online_20pct \
        --checkpoint-label 20pct --data-fraction 0.2 \
        --episode-list-json /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json \
        --raw-root /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw \
        --normalization /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json \
        --base-checkpoint /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base \
        --output /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/stage1_scaling_held_out/20pct.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-config", default="vtla_stage1_predictor_pretrain")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                         help="Exp-level checkpoint dir (e.g. .../stage1_online_20pct); "
                              "the LATEST step subdir found there is loaded. Omit with --baseline.")
    parser.add_argument("--baseline", action="store_true",
                         help="Score n0-vtla-base itself (no mid-train delta loaded): the released checkpoint's "
                              "own tactile encoder projection / predictor. Its recon head is untrained.")
    parser.add_argument("--checkpoint-label", required=True, help="e.g. '20pct' -- goes in the output JSON.")
    parser.add_argument("--data-fraction", type=float, required=True, help="e.g. 0.2 -- the x-axis value.")
    parser.add_argument("--episode-list-json", type=Path, required=True,
                         help="Manifest from build_itw_scaling_splits.py (must have a 'held_out' split).")
    parser.add_argument("--episode-list-key", default="held_out")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True,
                         help="n0-vtla-base dir or model.safetensors path (VTLA_PRETRAINED_CHECKPOINT).")
    parser.add_argument("--future-frame-offset", type=int, default=50)
    parser.add_argument("--default-prompt", default="Perform the task.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=200,
                         help="Cap on batches scored. ITWOnlineTactileDataset yields one sample per "
                              "FRAME within an episode, not one per episode -- a 'full epoch' over "
                              "~7700 held-out episodes is tens of thousands of batches (confirmed: "
                              "73841 at batch_size=64), not the ~120 a per-episode count would suggest. "
                              "200 batches (12800 frame-samples) gives a stable mean in well under an "
                              "hour instead of an eval that would never finish.")
    parser.add_argument("--contact-quantile", type=float, default=0.9,
                         help="Cells of the pooled target field above this quantile count as 'contact' "
                              "for the contact-region recon baselines.")
    parser.add_argument("--save-embeddings", type=Path, default=None,
                        help="Also save the pool's normalised (z, z*) as fp16 so retrieval can be recomputed offline.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.baseline == (args.checkpoint_dir is not None):
        raise ValueError("pass exactly one of --checkpoint-dir or --baseline")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_pytorch import setup_ddp  # noqa: E402
    from train_stage1_predictor import (  # noqa: E402
        STAGE1_TRAINABLE_PREFIXES,
        _Stage1Wrapper,
        load_stage1_checkpoint,
        load_stage1_policy_weights,
    )

    import n0vtla.training.itw_online_dataset as _online  # noqa: E402
    from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402
    from n0vtla.training.config import get_config  # noqa: E402

    config = get_config(args.stage1_config)
    model_cfg = config.model
    if not getattr(model_cfg, "stage1_pretrain_enabled", False):
        raise ValueError(f"{args.stage1_config!r} does not have stage1_pretrain_enabled=True")
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    manifest = json.loads(args.episode_list_json.read_text())
    try:
        rel_paths = manifest["splits"][args.episode_list_key]
    except KeyError as e:
        raise ValueError(f"key {args.episode_list_key!r} not found in {args.episode_list_json}'s splits "
                          f"(available: {sorted(manifest.get('splits', {}))})") from e
    episode_dirs = [args.raw_root / rel for rel in rel_paths]
    missing = [str(p) for p in episode_dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} held-out episode dirs missing under "
                                 f"{args.raw_root}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"Held-out set ({args.episode_list_key}): {len(episode_dirs)} episodes")

    device = torch.device(args.device)
    use_ddp, _local_rank, _device = setup_ddp()
    assert not use_ddp, "single-GPU eval only -- do not launch this via torchrun"

    loader = _online.create_stage1_data_loader(
        config, episode_dirs, str(args.normalization),
        future_frame_offset=args.future_frame_offset, default_prompt=args.default_prompt,
    )
    torch_loader = loader._data_loader.torch_loader
    epoch_batches = len(torch_loader)
    if epoch_batches == 0:
        raise ValueError("No complete batch available from the held-out set at this batch size")
    num_batches = min(epoch_batches, args.max_batches)
    print(f"Held-out set has {epoch_batches} batches of batch_size={config.batch_size} for one full "
          f"epoch (one sample per FRAME, not per episode); scoring the first {num_batches} of them.")

    policy = N0VTLAPolicy(model_cfg).to(device)
    base_weights = args.base_checkpoint / "model.safetensors" if args.base_checkpoint.is_dir() else args.base_checkpoint
    missing, unexpected = load_stage1_policy_weights(policy, base_weights, device, strict=False)
    allowed_missing = ("tactile_encoder.", "tactile_predictor.", "tactile_recon_head.", "z_proj.", "z_gate")
    missing_base = [name for name in missing if not name.startswith(allowed_missing)]
    if missing_base or unexpected:
        raise ValueError(f"Incompatible base checkpoint: missing_base={missing_base}, unexpected={unexpected}")

    for name, p in policy.named_parameters():
        p.requires_grad = name.startswith(STAGE1_TRAINABLE_PREFIXES)

    model = _Stage1Wrapper(policy)
    # load_stage1_checkpoint unconditionally loads optimizer state onto whatever
    # optimizer it's handed -- never stepped here, just satisfies that signature.
    dummy_optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-8)
    if args.baseline:
        loaded_step = -1
        print("BASELINE: n0-vtla-base only, no mid-train checkpoint loaded")
    else:
        loaded_step = load_stage1_checkpoint(model, dummy_optim, args.checkpoint_dir, device)
        print(f"Loaded checkpoint step {loaded_step} from {args.checkpoint_dir}")

    model.eval()
    infos = []
    import jax
    import time
    import torch.nn.functional as F

    # Capture (z, z*, valid) and the recon-head input/target by wrapping the policy's own
    # methods here, so the model code (shared with running training jobs) stays untouched.
    raw = model.policy
    cap: dict = {}
    orig_nce = raw._stage1_infonce_loss
    orig_bft = raw._build_future_target

    def nce_hook(z, z_star, valid):
        cap["z"], cap["z_star"], cap["valid"] = z.detach(), z_star.detach(), valid.detach()
        return orig_nce(z, z_star, valid)

    def bft_hook(*a, **k):
        out = orig_bft(*a, **k)
        cap["dbar_field"] = out[1].detach()
        return out

    raw._stage1_infonce_loss = nce_hook
    raw._build_future_target = bft_hook
    raw.tactile_recon_head.register_forward_hook(lambda m, i, o: cap.__setitem__("pred", o.detach()))
    grid = raw.tactile_recon_head.grid

    hz_all, hs_all, pred_all, tgt_all = [], [], [], []
    start = time.time()
    with torch.no_grad():
        for batch_idx, (observation, _actions) in enumerate(itertools.islice(loader, num_batches), start=1):
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            model(observation)
            info = dict(raw._last_loss_parts)
            if info.get("stage1_valid_count", 0) == 0:
                print(f"[{batch_idx}/{num_batches}] skipped (0 valid targets)", flush=True)
                continue
            infos.append(info)
            v = cap["valid"].bool()
            hz_all.append(F.normalize(cap["z"].float().mean(dim=1)[v], dim=-1))
            hs_all.append(F.normalize(cap["z_star"].float().mean(dim=1)[v], dim=-1))
            gray = cap["dbar_field"].float().mean(dim=1, keepdim=True)
            tgt_all.append(F.adaptive_avg_pool2d(gray, grid).squeeze(1)[v])
            pred_all.append(cap["pred"].float()[v])
            elapsed = time.time() - start
            rate = elapsed / batch_idx
            eta = rate * (num_batches - batch_idx)
            print(f"[{batch_idx}/{num_batches}] stage1_total={info['stage1_total']:.4f} "
                  f"elapsed={elapsed:.0f}s rate={rate:.2f}s/batch eta={eta:.0f}s", flush=True)

    if not infos:
        raise RuntimeError("Every held-out batch had zero valid future-tactile targets")

    total_valid = sum(i["stage1_valid_count"] for i in infos)
    weighted_means = {
        k: float(sum(i[k] * i["stage1_valid_count"] for i in infos) / total_valid)
        for k in ("stage1_nce", "stage1_recon", "stage1_total")
    }
    # Pool-wide retrieval: every valid sample's predicted latent (query) against ALL valid
    # real-future latents (gallery), not just its own batch's ~59 -- far more sensitive than
    # batch InfoNCE (temperature 1 + cosine logits compress that loss's dynamic range).
    hz, hs = torch.cat(hz_all), torch.cat(hs_all)
    n_pool = hz.shape[0]
    sim = hz @ hs.t()
    diag = sim.diagonal()
    rank_i2t = (sim > diag[:, None]).sum(1)
    rank_t2i = (sim.t() > diag[:, None]).sum(1)
    labels = torch.arange(n_pool, device=sim.device)
    retrieval = {
        "pool_size": int(n_pool),
        "chance_top1": 1.0 / n_pool,
        "pool_nce": float(0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))),
        "mean_pos_cos": float(diag.mean()),
        "mean_neg_cos": float((sim.sum() - diag.sum()) / (n_pool * n_pool - n_pool)),
    }
    # Tie-fair variant. Many samples (no tactile change) share IDENTICAL target latents, and the `>` rank
    # above ("optimistic ties") places the true partner FIRST among exact ties, inflating top-k. Here every
    # similarity gets tiny independent jitter, so the partner is placed uniformly among its ties
    # (averaged over several draws). Percentile rank counts ties as 0.5 (== per-query ROC-AUC).
    gen = torch.Generator(device=sim.device).manual_seed(0)
    n_draw = 3
    fair = {f"{n}_{k}": [] for n in ("i2t", "t2i") for k in ("top1", "top5", "top10", "top100", "mrr")}
    for _ in range(n_draw):
        sj = sim + 1e-5 * torch.randn(sim.shape, generator=gen, device=sim.device)
        dj = sj.diagonal()
        for name, r in (("i2t", (sj > dj[:, None]).sum(1)), ("t2i", (sj.t() > dj[:, None]).sum(1))):
            for k in (1, 5, 10, 100):
                fair[f"{name}_top{k}"].append(float((r < k).float().mean()))
            fair[f"{name}_mrr"].append(float((1.0 / (r.float() + 1)).mean()))
        del sj
    for k, v in fair.items():
        retrieval[f"{k}_fair"] = float(np.mean(v))
    retrieval["i2t_pct_rank"] = float((((sim < diag[:, None]).sum(1).float() + 0.5 * ((sim == diag[:, None]).sum(1).float() - 1))
                                       / (n_pool - 1)).mean())
    retrieval["n_unique_targets"] = int(torch.unique(hs.half(), dim=0).shape[0])
    for name, rank in (("i2t", rank_i2t), ("t2i", rank_t2i)):
        retrieval[f"{name}_top1"] = float((rank < 1).float().mean())
        retrieval[f"{name}_top5"] = float((rank < 5).float().mean())
        retrieval[f"{name}_top10"] = float((rank < 10).float().mean())
        retrieval[f"{name}_top100"] = float((rank < 100).float().mean())
        retrieval[f"{name}_mrr"] = float((1.0 / (rank.float() + 1)).mean())
        retrieval[f"{name}_median_rank"] = float(rank.float().median())
    del sim
    if args.save_embeddings is not None:
        args.save_embeddings.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hz": hz.half().cpu(), "hzs": hs.half().cpu(), "label": args.checkpoint_label}, args.save_embeddings)

    # Recon vs trivial baselines: the model's L1 only means something next to what predicting
    # zeros / the pool-mean field would score on the same 8x8 target.
    pred, tgt = torch.cat(pred_all), torch.cat(tgt_all)
    thr = float(torch.quantile(tgt.flatten()[:: max(1, tgt.numel() // 1_000_000)], args.contact_quantile))
    contact = tgt > thr
    mean_field = tgt.mean(dim=0, keepdim=True)
    scalar_mean = tgt.mean()

    def l1(a, b, mask=None):
        d = (a - b).abs()
        return float(d[mask].mean()) if mask is not None else float(d.mean())

    recon_baselines = {
        "target_mean": float(tgt.mean()), "target_std": float(tgt.std()), "target_max": float(tgt.max()),
        "contact_threshold": thr, "contact_quantile": args.contact_quantile,
        "contact_cell_frac": float(contact.float().mean()),
        "l1_model": l1(pred, tgt), "l1_zero": l1(torch.zeros_like(tgt), tgt),
        "l1_scalar_mean": l1(scalar_mean.expand_as(tgt), tgt), "l1_pool_mean_field": l1(mean_field.expand_as(tgt), tgt),
        "contact_l1_model": l1(pred, tgt, contact), "contact_l1_zero": l1(torch.zeros_like(tgt), tgt, contact),
        "contact_l1_pool_mean_field": l1(mean_field.expand_as(tgt), tgt, contact),
    }
    recon_baselines["skill_vs_pool_mean_field"] = 1.0 - recon_baselines["l1_model"] / recon_baselines["l1_pool_mean_field"]
    recon_baselines["contact_skill_vs_pool_mean_field"] = (
        1.0 - recon_baselines["contact_l1_model"] / recon_baselines["contact_l1_pool_mean_field"]
    )

    report = {
        "retrieval": retrieval,
        "recon_baselines": recon_baselines,
        "checkpoint_label": args.checkpoint_label,
        "data_fraction": args.data_fraction,
        "checkpoint_step": loaded_step,
        "episode_list_key": args.episode_list_key,
        "num_held_out_episodes": len(episode_dirs),
        "epoch_batches_available": epoch_batches,
        "num_batches_scored": len(infos),
        "num_batches_skipped_empty": num_batches - len(infos),
        "total_valid_samples": int(total_valid),
        "mean_stage1_valid_frac": float(np.mean([i["stage1_valid_frac"] for i in infos])),
        **{f"mean_{k}": v for k, v in weighted_means.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
