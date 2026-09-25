#!/usr/bin/env python
"""Collect per-sample PREDICTIONS of several mid-train (Stage-1) checkpoints in ONE pass, so
retrieval / contact-detection metrics can be recomputed offline (CPU) with any protocol.

Why one pass for all checkpoints: the merged mid-train checkpoints differ from n0-vtla-base ONLY
in `tactile_encoder.tactile_proj.*`, `tactile_predictor.*` (+ the Stage-1-only recon head), so the
frozen ~3.7B VLM prefix forward is identical for all of them. We run it once per batch and only swap
those ~123M parameters per checkpoint. Every checkpoint therefore sees the identical samples, the
identical (deterministic, no-augmentation) images, and the identical DINOv2 features.

Per valid sample it stores (fp16; K = number of checkpoints, D = llm_dim):
  hz[K,N,D]     token-mean-pooled, L2-normalised predicted latent z          (the retrieval query)
  hzs[K,N,D]    same for the real future latent z* (ckpt's OWN tactile encoder)   (the target)
  hcur[K,N,D]   pooled/normalised embedding of the CURRENT tactile diff (tac_t - tac_0), i.e. the
                predictor's own input g  -> "persistence" baseline, in the ckpt's own space
  f_star[N,768], f_cur[N,768]   frozen-DINOv2 features (mean over tokens/views, BEFORE the trainable
                projection) of the future / current diff: a checkpoint-independent target space
  grid_tgt[N,8,8]   8x8 average-pooled (tac_{t+H} - tac_t) field (channel mean)   [physical target]
  grid_pred[K,N,8,8] recon head prediction from z (NaN for `base`, which has no trained head)
  d_*, cur_mean_abs, fut_mean_abs  scalar contact-change / contact-state statistics
  episode_index, frame_in_episode, row
Sampling is deterministic (fixed stride). VTLA_DATASET_PATH / VTLA_ASSET_ID must be exported
before launch (config.py reads them at import); norm stats at
assets/vtla_stage1_predictor_pretrain/<asset_id>/norm_stats.json.

  python scripts/collect_stage1_multi_ckpt_predictions.py --base <n0-vtla-base dir> \
      --ckpt 20pct=<delta dir> --ckpt 40pct=<delta dir> ... --dataset-root $VTLA_DATASET_PATH \
      --split-label val --stride 2 --out out/task1_val.pt
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-config", default="vtla_stage1_predictor_pretrain")
    p.add_argument("--base", type=Path, required=True, help="dir with the FULL n0-vtla-base model.safetensors")
    p.add_argument("--ckpt", action="append", default=[], metavar="LABEL=DELTA_DIR",
                   help="mid-train checkpoint: label=dir holding the trainable-only delta model.safetensors")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split-label", required=True)
    p.add_argument("--task-label", default="")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-preprocess", action="store_true",
                   help="use train=True image preprocessing (random aug) like the v3 script; default: deterministic")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(args.out)
    for var in ("VTLA_ASSET_ID", "VTLA_DATASET_PATH"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} must be exported before running")
    if Path(os.environ["VTLA_DATASET_PATH"]).resolve() != args.dataset_root.resolve():
        raise RuntimeError("VTLA_DATASET_PATH != --dataset-root")

    import jax
    import safetensors.torch
    import torch
    import torch.nn.functional as F
    import torch.utils.data as tud

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    device = torch.device(args.device)
    config = _config.get_config(args.stage1_config)
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    data_config = config.data.create(config.assets_dirs, model_cfg)
    norm_path = config.assets_dirs / str(data_config.asset_id) / "norm_stats.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"norm stats missing at {norm_path}")

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = np.array([ep["length"] for ep in episodes])
    boundaries = np.concatenate([[0], np.cumsum(lengths)])
    total_rows = int(boundaries[-1])
    dataset = _data.create_torch_dataset(data_config, model_cfg.action_horizon, model_cfg)
    dataset = _data.transform_dataset(dataset, data_config)
    if len(dataset) != total_rows:
        raise RuntimeError(f"dataset length {len(dataset)} != episodes.jsonl total {total_rows}")

    idx = np.arange(0, total_rows, args.stride)
    if args.max_samples and len(idx) > args.max_samples:
        idx = idx[np.unique(np.linspace(0, len(idx) - 1, args.max_samples).astype(int))]
    ep_of = np.searchsorted(boundaries, idx, side="right") - 1
    frame_in_ep = idx - boundaries[ep_of]
    ep_index = np.array([episodes[e]["episode_index"] for e in ep_of])
    print(f"{args.task_label}/{args.split_label}: {len(episodes)} episodes, {total_rows} frames -> {len(idx)} sampled", flush=True)

    workers = args.num_workers
    loader = tud.DataLoader(
        tud.Subset(dataset, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=workers,
        multiprocessing_context=multiprocessing.get_context("spawn") if workers > 0 else None,
        persistent_workers=workers > 0, collate_fn=_data._collate_fn, worker_init_fn=_data._worker_init_fn,
        drop_last=False,
    )

    print(f"=== loading base {args.base} ===", flush=True)
    model = N0VTLAPolicy(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(model, str(args.base / "model.safetensors"), strict=False, device=str(device))
    allowed_missing = ("tactile_recon_head.", "z_proj.", "z_gate")
    bad = [k for k in missing if not k.startswith(allowed_missing)]
    if bad:
        raise RuntimeError(f"base checkpoint missing required keys: {bad[:10]}")
    model.eval()

    deltas: dict[str, dict] = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        sd = safetensors.torch.load_file(str(Path(path) / "model.safetensors"), device=str(device))
        bad = [k for k in sd if not k.startswith(("tactile_encoder.tactile_proj.", "tactile_predictor.", "tactile_recon_head."))]
        if bad:
            raise RuntimeError(f"{label}: unexpected keys in delta: {bad[:5]}")
        deltas[label] = sd
    swap_keys = sorted({k for sd in deltas.values() for k in sd})
    full = model.state_dict()
    base_state = {k: full[k].detach().clone() for k in swap_keys}
    labels = ["base"] + list(deltas)
    K = len(labels)
    print(f"checkpoints: {labels}; swapping {len(swap_keys)} tensors per checkpoint", flush=True)

    def activate(label: str) -> None:
        model.load_state_dict(base_state if label == "base" else deltas[label], strict=False)

    def masked_mean_views(per_view: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        st = torch.stack(per_view, 0)                     # (V, B, ...)
        w = torch.stack(masks, 0).float()                 # (V, B)
        w = w / w.sum(0, keepdim=True).clamp(min=1.0)
        return (st * w.view(*w.shape, *([1] * (st.dim() - 2)))).sum(0)

    acc = {k: [] for k in ("hz", "hzs", "hcur", "grid_pred")}
    fixed = {k: [] for k in ("f_star", "f_cur", "grid_tgt", "d_mean_abs", "d_max_abs", "d_l2", "d_pos", "d_neg",
                             "cur_mean_abs", "fut_mean_abs")}
    keep_pos = []
    pos = 0
    start = time.time()
    with torch.no_grad():
        for b, batch in enumerate(loader, start=1):
            batch = jax.tree.map(torch.as_tensor, batch)
            observation = _data._model.Observation.from_dict(batch)
            observation = jax.tree.map(lambda x: x.to(device), observation)
            images, img_masks, lang_tokens, lang_masks, _s, _e = model._preprocess_observation(
                observation, train=args.train_preprocess)
            vl_ctx, _pe, prefix_pad_masks, _pam, _pkv = model._prefix_forward(
                images, img_masks, lang_tokens, lang_masks, use_cache=False)
            tac_t, tac_0, tac_f = model._last_tac_t or {}, model._last_tac_0 or {}, model._last_tac_f or {}
            vm, vmf = model._last_tac_mask or {}, model._last_tac_mask_f or {}

            per_z, per_zs, per_cur, per_pred = [], [], [], []
            valid = None
            for label in labels:
                activate(label)
                z, g, has_tac = model._compute_z(vl_ctx, prefix_pad_masks)
                z_star, dbar, has_future = model._build_future_target(tac_f, tac_t, getattr(model, "_last_tac_mask_f", None))
                if z_star is None:
                    raise RuntimeError("no future tactile frame (future_frame_offset must be > 0)")
                v = (has_tac & has_future).bool().to(device)
                valid = v if valid is None else valid
                gm = model._last_g_mask.to(device).float()          # (B, tokens*V)
                gcur = (g.float() * gm[..., None]).sum(1) / gm.sum(1, keepdim=True).clamp(min=1.0)
                per_z.append(F.normalize(z.float().mean(dim=1), dim=-1))
                per_zs.append(F.normalize(z_star.float().mean(dim=1), dim=-1))
                per_cur.append(F.normalize(gcur, dim=-1))
                if label == "base":
                    per_pred.append(torch.full((z.shape[0], 8, 8), float("nan"), device=device))
                else:
                    per_pred.append(model.tactile_recon_head(z).float())
            # checkpoint-independent quantities
            keys = [k for k in tac_f if k in tac_t]
            d_views, m_views, fs_views, fc_views, cur_views, fut_views = [], [], [], [], [], []
            for k in keys:
                dfut = tac_f[k].float() - tac_t[k].float()
                d_views.append(dfut)
                mk = vmf.get(k)
                mk = torch.ones(dfut.shape[0], dtype=torch.bool, device=device) if mk is None else mk.to(device).bool().reshape(-1)
                cm = vm.get(k)
                if cm is not None:
                    mk = mk & cm.to(device).bool().reshape(-1)
                m_views.append(mk)
                enc = model.tactile_encoder
                fs_views.append(enc.encode_backbone(dfut).float().mean(1))
                if k in tac_0:
                    dcur = tac_t[k].float() - tac_0[k].float()
                    fc_views.append(enc.encode_backbone(dcur).float().mean(1))
                    cur_views.append(dcur.abs().flatten(1).mean(1))
                    fut_views.append((tac_f[k].float() - tac_0[k].float()).abs().flatten(1).mean(1))
                else:
                    fc_views.append(torch.zeros_like(fs_views[-1]))
                    cur_views.append(torch.zeros(dfut.shape[0], device=device))
                    fut_views.append(torch.zeros(dfut.shape[0], device=device))
            dbar = masked_mean_views(d_views, m_views)                       # (B, C, H, W)
            f_star = masked_mean_views(fs_views, m_views)
            f_cur = masked_mean_views(fc_views, m_views)
            cur_abs = masked_mean_views(cur_views, m_views)
            fut_abs = masked_mean_views(fut_views, m_views)
            grid = F.adaptive_avg_pool2d(dbar.mean(dim=1, keepdim=True), 8).squeeze(1)
            flat = dbar.flatten(1)

            vc = valid.cpu()
            sel = lambda t: t.detach().cpu()[vc]  # noqa: E731
            acc["hz"].append(torch.stack([sel(t) for t in per_z], 0).half())
            acc["hzs"].append(torch.stack([sel(t) for t in per_zs], 0).half())
            acc["hcur"].append(torch.stack([sel(t) for t in per_cur], 0).half())
            acc["grid_pred"].append(torch.stack([sel(t) for t in per_pred], 0).half())
            fixed["f_star"].append(sel(f_star).half())
            fixed["f_cur"].append(sel(f_cur).half())
            fixed["grid_tgt"].append(sel(grid).half())
            fixed["d_mean_abs"].append(sel(flat.abs().mean(1)))
            fixed["d_max_abs"].append(sel(flat.abs().max(1).values))
            fixed["d_l2"].append(sel(flat.norm(dim=1) / flat.shape[1] ** 0.5))
            fixed["d_pos"].append(sel(F.relu(flat).mean(1)))
            fixed["d_neg"].append(sel(F.relu(-flat).mean(1)))
            fixed["cur_mean_abs"].append(sel(cur_abs))
            fixed["fut_mean_abs"].append(sel(fut_abs))
            keep_pos.append(pos + torch.nonzero(vc).flatten().numpy())
            pos += int(valid.shape[0])
            if b % 20 == 0 or b == len(loader):
                el = time.time() - start
                print(f"[{b}/{len(loader)}] samples={pos} valid={sum(len(k) for k in keep_pos)} "
                      f"elapsed={el:.0f}s eta={el / b * (len(loader) - b):.0f}s", flush=True)

    keep = np.concatenate(keep_pos)
    out = {
        "labels": labels, "task": args.task_label, "split": args.split_label,
        "hz": torch.cat(acc["hz"], 1), "hzs": torch.cat(acc["hzs"], 1), "hcur": torch.cat(acc["hcur"], 1),
        "grid_pred": torch.cat(acc["grid_pred"], 1),
        **{k: torch.cat(v) for k, v in fixed.items()},
        "episode_index": torch.as_tensor(ep_index[keep]), "frame_in_episode": torch.as_tensor(frame_in_ep[keep]),
        "row": torch.as_tensor(idx[keep]), "future_frame_offset": int(getattr(data_config, "future_frame_offset", 50) or 50),
        "num_sampled": int(len(idx)), "stride": args.stride, "n_episodes": len(episodes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Wrote {args.out}: K={K} N={len(keep)} D={out['hz'].shape[-1]}", flush=True)


if __name__ == "__main__":
    main()
