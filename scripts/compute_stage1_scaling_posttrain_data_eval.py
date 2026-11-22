#!/usr/bin/env python
"""Stage-1 checkpoints scored on POST-TRAIN (robot wetlab) data: predicted future-tactile
latent z vs the real future latent z*, as pool-wide retrieval.

WHY: the Stage-1 scaling checkpoints were trained on human ITW data only, so the post-train
task's train AND val splits are both unseen by them -- a cross-embodiment transfer readout
in the domain the post-train actually uses. Reported per split (val is small and is the
only split post-train never sees either), and the caller can pool train+val downstream.

This script only COLLECTS embeddings: for each strided, valid sample it stores the L2-
normalised, token-mean-pooled z and z* (fp16) plus episode/frame ids. Metrics (retrieval
top-k / MRR / pos-neg cosine, per-split and pooled, episode-clustered bootstrap CIs) are
computed on CPU by aggregate_stage1_scaling_posttrain_data_eval.py so several checkpoints
and splits can be pooled without re-running the model.

The checkpoint must be a COMPLETE model dir (merged base+Stage-1 delta, or n0-vtla-base
itself as the "no Stage-1" reference) -- see merge_stage1_into_base_checkpoint.py.
Sampling is deterministic (sequential index order, fixed stride), so every checkpoint sees
the identical sample set. Model code is reused unchanged: same _preprocess_observation /
_prefix_forward / _compute_z / _build_future_target path as Stage-1 training.

VTLA_DATASET_PATH and VTLA_ASSET_ID must be exported BEFORE launching (config.py reads them
at import), and norm stats must exist at
assets/vtla_stage1_predictor_pretrain/<asset_id>/norm_stats.json.

Usage:
    export VTLA_DATASET_PATH=.../canonical_wetlab_task1_val VTLA_ASSET_ID=task1_tuberack_train
    python scripts/compute_stage1_scaling_posttrain_data_eval.py \
        --checkpoint .../n0-vtla-base_plus_stage1_scaling_20pct_14000 --checkpoint-label 20pct \
        --dataset-root $VTLA_DATASET_PATH --split-label val \
        --output out/20pct_val.json --embeddings-out out/20pct_val.pt
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-config", default="vtla_stage1_predictor_pretrain",
                         help="TrainConfig used only to build the model skeleton + the canonical tactile "
                              "data pipeline with a future frame; weights come from --checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="Dir holding a COMPLETE model.safetensors (merged ckpt, or n0-vtla-base).")
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-label", required=True, help="e.g. train / val")
    parser.add_argument("--stride", type=int, default=5, help="Take every stride-th frame of the split.")
    parser.add_argument("--max-samples", type=int, default=0,
                         help="If >0, keep at most this many strided samples, evenly spread over the split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embeddings-out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for p in (args.output, args.embeddings_out):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"{p} exists; pass --overwrite")
    for var in ("VTLA_ASSET_ID", "VTLA_DATASET_PATH"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} must be exported before running (config.py reads it at import)")
    if Path(os.environ["VTLA_DATASET_PATH"]).resolve() != args.dataset_root.resolve():
        raise RuntimeError(f"VTLA_DATASET_PATH={os.environ['VTLA_DATASET_PATH']} != --dataset-root {args.dataset_root}")

    import safetensors.torch
    import torch
    import torch.nn.functional as F
    import torch.utils.data as tud
    import jax

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
        raise FileNotFoundError(f"norm stats missing at {norm_path} (copy the task's norm_stats.json there)")
    print(f"norm stats: {norm_path}")

    episodes = [json.loads(line) for line in (args.dataset_root / "meta" / "episodes.jsonl").open()]
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
    print(f"split={args.split_label}: {len(episodes)} episodes, {total_rows} frames -> {len(idx)} sampled "
          f"(stride {args.stride}, cap {args.max_samples or 'none'})")

    workers = args.num_workers
    loader = tud.DataLoader(
        tud.Subset(dataset, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=workers,
        multiprocessing_context=multiprocessing.get_context("spawn") if workers > 0 else None,
        persistent_workers=workers > 0, collate_fn=_data._collate_fn, worker_init_fn=_data._worker_init_fn,
        drop_last=False,
    )

    print(f"=== loading {args.checkpoint_label}: {args.checkpoint} ===")
    model = N0VTLAPolicy(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(
        model, str(args.checkpoint / "model.safetensors"), strict=False, device=str(device)
    )
    # tactile_recon_head exists only in the Stage-1 skeleton and is dropped by the merge;
    # z_proj/z_gate are not used by the (z, z*) retrieval below. Anything else missing (VLM,
    # tactile_encoder, tactile_predictor) would silently leave random weights -- refuse.
    allowed_missing = ("tactile_recon_head.", "z_proj.", "z_gate")
    bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
    if bad_missing:
        raise RuntimeError(f"checkpoint is missing required keys (would stay random-init): {bad_missing[:10]}")
    print(f"  missing (allowed): {len(missing)} keys; unexpected (ignored): {list(unexpected)[:5]}")
    model.eval()
    torch.manual_seed(args.seed)

    hz_list, hs_list, keep_pos = [], [], []
    n_no_target = 0
    start = time.time()
    pos = 0
    with torch.no_grad():
        for b, batch in enumerate(loader, start=1):
            batch = jax.tree.map(torch.as_tensor, batch)
            observation = _data._model.Observation.from_dict(batch)
            observation = jax.tree.map(lambda x: x.to(device), observation)
            images, img_masks, lang_tokens, lang_masks, _s, _e = model._preprocess_observation(
                observation, train=True
            )
            vl_ctx, _pe, prefix_pad_masks, _pam, _pkv = model._prefix_forward(
                images, img_masks, lang_tokens, lang_masks, use_cache=False
            )
            z, _g, has_tac = model._compute_z(vl_ctx, prefix_pad_masks)
            z_star, _dbar, has_future = model._build_future_target(
                model._last_tac_f or {}, model._last_tac_t or {}, getattr(model, "_last_tac_mask_f", None)
            )
            if z_star is None:
                raise RuntimeError("no '.future' tactile frame in the batch (future_frame_offset must be >0)")
            valid = (has_tac & has_future).bool()
            n = int(valid.shape[0])
            n_no_target += int((~valid).sum())
            hz_list.append(F.normalize(z.float().mean(dim=1)[valid], dim=-1).half().cpu())
            hs_list.append(F.normalize(z_star.detach().float().mean(dim=1)[valid], dim=-1).half().cpu())
            keep_pos.append(pos + torch.nonzero(valid.cpu()).flatten().numpy())
            pos += n
            if b % 20 == 0 or b == len(loader):
                el = time.time() - start
                print(f"[{b}/{len(loader)}] samples={pos} valid={sum(len(k) for k in keep_pos)} "
                      f"elapsed={el:.0f}s eta={el / b * (len(loader) - b):.0f}s", flush=True)

    keep = np.concatenate(keep_pos)
    hz, hs = torch.cat(hz_list), torch.cat(hs_list)
    if len(keep) < 2:
        raise RuntimeError(f"only {len(keep)} valid samples")
    args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "hz": hz, "hzs": hs,
        "episode_index": torch.as_tensor(ep_index[keep]), "frame_in_episode": torch.as_tensor(frame_in_ep[keep]),
        "checkpoint_label": args.checkpoint_label, "split": args.split_label,
    }, args.embeddings_out)
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_label": args.checkpoint_label,
        "split": args.split_label, "dataset_root": str(args.dataset_root),
        "num_episodes": len(episodes), "total_frames": total_rows, "stride": args.stride,
        "max_samples": args.max_samples, "num_sampled": int(len(idx)), "num_valid": int(len(keep)),
        "num_no_future_target": n_no_target, "num_episodes_with_valid": int(len(np.unique(ep_index[keep]))),
        "embedding_dim": int(hz.shape[1]), "embeddings_file": str(args.embeddings_out),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
