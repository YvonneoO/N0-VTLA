#!/usr/bin/env python
"""Stage-2 latent-to-expert alignment (paper arXiv:2607.23782 Sec 4.2, "Stage 2: aligning
latents with the action expert") -- a SMALL-SCALE bridge, not a full paper-faithful
re-alignment run (see docs/PRETRAIN_IMPLEMENTATION.md and this repo's design discussion for
why: n0-vtla-base's action expert was already taught to consume z by NeoteAI's own Stage 2/3;
our Stage-1 human-data run only continues fine-tuning an already-integrated interface, and its
own NCE-loss trajectory shows z's discriminative geometry has not drifted far from what the
action expert already expects. This script exists as a cheap fallback in case that shift is
larger than the NCE trend suggests -- if it turns out to be a no-op, that itself is a valid
finding).

Paper text (Sec 4.2, Stage 2): "We hold the tactile perception stack frozen at its Stage 1
checkpoint and train only the latent-to-expert projection and the action expert, under the
base action objective. Concretely, in the expert's attention the keys and values from the
vision-language prefix are masked out for the action queries, leaving the latent tokens z and
the noised action tokens as the only conditioning the expert can attend to."

Implementation notes (see STAGE2_TRAINABLE_PREFIXES below for the exact freeze set):
  * The "base action objective" is exactly N0VTLAPolicy._forward_predictor's flow-matching MSE
    -- reused unchanged via the normal `policy(observation, actions)` forward, no new loss code.
  * The prefix-masking behavior is exactly `N0VTLAConfig.vl_dropout_prob=1.0` (see
    N0VTLAPolicy._vl_dropout_keep): at p=1.0, every sample's suffix (z + noised action tokens)
    is deterministically cut off from all vision-language prefix key/value attention, while z's
    own computation (which happens before this mask) is untouched. No new masking code.
  * "the tactile perception stack frozen at its Stage 1 checkpoint" means TWO warm-start steps,
    unconditional every launch (resume included), mirroring train_stage1_predictor.py's own
    pattern: (1) the base policy checkpoint (e.g. n0-vtla-base) via load_stage1_policy_weights,
    then (2) Stage 1's own trainable-only checkpoint on top via load_stage1_tactile_delta below
    (which additionally tolerates tactile_recon_head.* being present in that checkpoint but
    absent from this stage's model, since Stage 1's reconstruction head is Sec-4.2-Stage-1-only
    and this config never sets stage1_pretrain_enabled).

Usage (single node):
    torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \\
        scripts/train_stage2_align_expert.py vtla_stage2_align_expert \\
        --exp-name=my_stage2_run

Env vars (same convention as train_stage1_predictor.py):
    VTLA_PRETRAINED_CHECKPOINT   required. Base policy checkpoint dir (e.g. n0-vtla-base).
    VTLA_STAGE1_CHECKPOINT       required. Stage-1's own checkpoint dir to load the tactile
                                 delta from, e.g.
                                 checkpoints/vtla_stage1_predictor_pretrain/online_full_v3
                                 (the latest numeric step subdir under it is used).
    VTLA_DATASET_PATH            required. Local canonical LeRobot dataset (see
                                 scripts/download_openneodata_flexiv_smoke.py).
    VTLA_ASSET_ID                norm-stats asset id (see scripts/compute_canonical_norm.py).
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pytorch import (  # noqa: E402
    get_latest_checkpoint_step,
    init_logging,
    init_wandb,
    set_seed,
    setup_ddp,
    wait_for_path_state,
)
from train_stage1_predictor import load_stage1_policy_weights  # noqa: E402

# The n0vtla package is installed editable, pinned to a fixed on-disk path (the main repo
# checkout) at env-creation time. When this script runs from an isolated `git worktree` (as it
# must, to avoid touching the main checkout while another training chain reads/writes it -- see
# this project's Stage-2 worktree workflow), a bare `import n0vtla` run as a *script* (not `-c`,
# where sys.path[0] would be cwd) resolves sys.path[0] to this file's own directory (scripts/),
# NOT the worktree root -- so it silently falls through to the editable install and imports the
# STALE, main-checkout copy of n0vtla instead of this worktree's own copy. That stale copy is
# missing this script's own vtla_stage2_align_expert config entry (added only on this branch),
# which surfaces as tyro reporting "vtla_stage2_align_expert" as an unrecognized subcommand even
# though it's plainly defined in n0vtla/training/config.py -- confusing unless you know to check
# `n0vtla.__file__`. Force the worktree root onto sys.path ahead of the stale editable install so
# every n0vtla submodule below resolves to this worktree's own code.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import n0vtla.training.config as _config  # noqa: E402
import n0vtla.training.data_loader as _data  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402


class _Stage2Wrapper(nn.Module):
    """Same rationale as train_stage1_predictor.py's _Stage1Wrapper: DDP's forward hooks are
    wired to DistributedDataParallel.forward specifically, so route through this module's own
    forward() rather than calling a differently-named method on a DDP-wrapped model."""

    def __init__(self, policy: N0VTLAPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation, actions):
        return self.policy(observation, actions)


# "the latent-to-expert projection and the action expert" (paper). action_in_proj/
# action_out_proj/time_mlp_in/time_mlp_out are the expert's action/time I/O -- always built
# for pi05=True (every real config in this repo) and meaningless without the expert transformer
# they feed, so they count as part of "the action expert" alongside
# paligemma_with_expert.gemma_expert itself. Everything else (paligemma_with_expert.paligemma.*,
# tactile_encoder.*, tactile_predictor.*) stays frozen at its Stage-1/base value.
STAGE2_TRAINABLE_PREFIXES = (
    "z_proj.",
    "z_gate",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "paligemma_with_expert.gemma_expert.",
)

# Stage-1's own trainable set (must match scripts/train_stage1_predictor.py's constant of the
# same values -- duplicated here, not imported, to keep this script's warm-start logic readable
# without requiring readers to cross-reference two files for what it loads).
_STAGE1_TACTILE_PREFIXES = ("tactile_encoder.tactile_proj.", "tactile_predictor.")
_STAGE1_RECON_HEAD_PREFIX = "tactile_recon_head."


def load_stage1_tactile_delta(policy: N0VTLAPolicy, stage1_checkpoint_dir: Path, device) -> int:
    """Loads Stage 1's trainable-only checkpoint (tactile_encoder.tactile_proj + tactile_predictor
    + tactile_recon_head, see save_stage1_checkpoint in train_stage1_predictor.py) ON TOP of the
    already-loaded base policy, EXCEPT tactile_recon_head -- Stage 2's model never constructs
    that submodule (stage1_pretrain_enabled=False here), so those keys are expected-but-unused,
    not an error. Returns the Stage-1 global_step this delta came from, for logging only (Stage 2
    tracks its OWN global_step separately via save_stage2_checkpoint/load_stage2_checkpoint).
    """
    steps = [
        int(d.name) for d in stage1_checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No Stage-1 checkpoints found in {stage1_checkpoint_dir}")
    latest = max(steps)
    ckpt_dir = stage1_checkpoint_dir / f"{latest}"
    missing, unexpected = safetensors.torch.load_model(
        policy, ckpt_dir / "model.safetensors", strict=False, device=str(device)
    )
    # `unexpected` = keys present in Stage-1's checkpoint file but not in this (Stage-2) model's
    # state_dict; tactile_recon_head.* is the ONLY expected case (Stage 1's auxiliary head,
    # never constructed here) -- anything else unexpected is a real mismatch.
    unexpected_real = [n for n in unexpected if not n.startswith(_STAGE1_RECON_HEAD_PREFIX)]
    if unexpected_real:
        raise ValueError(f"Unexpected keys loading Stage-1 checkpoint {ckpt_dir}: {unexpected_real}")
    missing_tactile = [n for n in missing if n.startswith(_STAGE1_TACTILE_PREFIXES)]
    if missing_tactile:
        raise ValueError(f"Stage-1 checkpoint {ckpt_dir} is missing tactile params: {missing_tactile}")
    return latest


def save_stage2_checkpoint(model, optimizer, global_step, config, is_main):
    """Trainable-only Stage-2 checkpoint (z_proj/z_gate/action expert I/O + the expert
    transformer itself -- STAGE2_TRAINABLE_PREFIXES), NOT a full policy snapshot. Mirrors
    save_stage1_checkpoint's format/rationale exactly (see train_stage1_predictor.py): the
    frozen base + Stage-1 tactile delta never change during Stage 2 and are reloaded fresh from
    disk on every launch (resume included), so re-saving them every checkpoint would be pure
    waste.
    """
    if not is_main:
        return
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or (
        global_step == config.num_train_steps
    )
    if not should_save:
        return
    final_dir = config.checkpoint_dir / f"{global_step}"
    tmp_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module.policy if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.policy
    trainable_state = {
        name: param.detach().to("cpu").contiguous()
        for name, param in raw_model.named_parameters()
        if name.startswith(STAGE2_TRAINABLE_PREFIXES)
    }
    safetensors.torch.save_file(trainable_state, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {"global_step": global_step, "step_format": "completed_updates",
         "config": dataclasses.asdict(config), "timestamp": time.time(),
         "checkpoint_format": "trainable_only_v1"},
        tmp_dir / "metadata.pt",
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved stage-2 checkpoint (trainable-only, {len(trainable_state)} tensors) at step {global_step} -> {final_dir}")


def load_stage2_checkpoint(model, optimizer, checkpoint_dir, device):
    """Resumes STAGE 2's OWN training (distinct from load_stage1_tactile_delta, which is a
    one-time warm-start from a DIFFERENT stage's checkpoint dir). Mirrors
    train_stage1_predictor.py's load_stage1_checkpoint exactly, filtered by
    STAGE2_TRAINABLE_PREFIXES instead."""
    steps = [
        int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(steps)
    ckpt_dir = checkpoint_dir / f"{latest}"
    raw_model = model.module.policy if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.policy
    missing, unexpected = safetensors.torch.load_model(
        raw_model, ckpt_dir / "model.safetensors", strict=False, device=str(device)
    )
    if unexpected:
        raise ValueError(f"Unexpected keys in trainable-only checkpoint {ckpt_dir}: {unexpected}")
    missing_trainable = [name for name in missing if name.startswith(STAGE2_TRAINABLE_PREFIXES)]
    if missing_trainable:
        raise ValueError(f"Trainable-only checkpoint {ckpt_dir} is missing trainable params: {missing_trainable}")
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=False))
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
    step = metadata.get("global_step", latest)
    return step if metadata.get("step_format") == "completed_updates" else step + 1


def train_loop_stage2(config: _config.TrainConfig) -> None:
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    global_rank = dist.get_rank() if use_ddp else 0
    set_seed(config.seed, global_rank)

    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    if not model_cfg.tactile_predictor_enabled or model_cfg.tactile_mode != "latent":
        raise ValueError("Stage 2 requires the latent tactile predictor")
    if getattr(model_cfg, "stage1_pretrain_enabled", False):
        raise ValueError("Stage 2 must NOT set stage1_pretrain_enabled (no NCE/recon targets)")
    if float(getattr(model_cfg, "vl_dropout_prob", 0.0)) != 1.0:
        raise ValueError("Stage 2 requires vl_dropout_prob=1.0 (mask VL prefix from the expert)")
    if not config.pytorch_weight_path:
        raise ValueError("Stage 2 requires a pretrained base policy checkpoint")
    ckpt_path = Path(config.pytorch_weight_path)
    weights_file = ckpt_path / "model.safetensors" if ckpt_path.is_dir() else ckpt_path
    if not weights_file.is_file():
        raise FileNotFoundError(f"Pretrained base policy checkpoint missing: {weights_file}")
    stage1_ckpt_env = os.environ.get("VTLA_STAGE1_CHECKPOINT")
    if not stage1_ckpt_env:
        raise ValueError("Stage 2 requires VTLA_STAGE1_CHECKPOINT (Stage-1's checkpoint dir)")
    stage1_checkpoint_dir = Path(stage1_ckpt_env)
    if not stage1_checkpoint_dir.is_dir():
        raise FileNotFoundError(f"VTLA_STAGE1_CHECKPOINT does not exist: {stage1_checkpoint_dir}")

    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(config.checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(f"Resuming stage-2 run from step {latest_step}")
            else:
                raise FileNotFoundError(f"No valid checkpoints in {config.checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"{config.checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
    elif config.checkpoint_dir.exists():
        raise FileExistsError(f"{config.checkpoint_dir} exists; use --resume or a new exp-name")

    if use_ddp:
        dist.barrier()

    if not resuming:
        ready_file = config.checkpoint_dir / ".ddp_fs_ready"
        if is_main:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ready_file.write_text("ready\n", encoding="utf-8")
        elif use_ddp:
            wait_for_path_state(ready_file, should_exist=True)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size < 2 or config.batch_size % world_size:
        raise ValueError("Stage-2 global batch_size must be >=2 and divisible by world_size")
    logging.info(f"stage2: batch_size={config.batch_size} (per-GPU={config.batch_size // world_size}) world_size={world_size}")

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    torch_loader = loader._data_loader.torch_loader
    batches_per_epoch = len(torch_loader)
    if batches_per_epoch == 0:
        raise ValueError("No complete batch available for Stage 2")
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        logging.warning(
            "Stage 2 is a SMALL-SCALE interface bridge (see this script's module docstring for "
            "why it is not a full paper-faithful re-alignment run), not paper Sec 4.2's original "
            "large-scale Stage 2. vl_dropout_prob=%s, z_gate_zero_init=%s.",
            model_cfg.vl_dropout_prob, getattr(model_cfg, "z_gate_zero_init", False),
        )

    policy = N0VTLAPolicy(model_cfg).to(device)

    # Warm-start, unconditional every launch (resume included) -- trainable-only checkpoints
    # don't carry the frozen base, so both the base AND the Stage-1 tactile delta are reloaded
    # fresh every time, with a resumed run's own Stage-2 checkpoint applied on top afterward.
    missing, unexpected = load_stage1_policy_weights(policy, weights_file, device, strict=False)
    allowed_missing = ("tactile_encoder.", "tactile_predictor.", "tactile_recon_head.", "z_proj.", "z_gate")
    missing_base = [name for name in missing if not name.startswith(allowed_missing)]
    if missing_base or unexpected:
        raise ValueError(f"Incompatible pretrained checkpoint: missing_base={missing_base}, unexpected={unexpected}")
    stage1_step = load_stage1_tactile_delta(policy, stage1_checkpoint_dir, device)
    if is_main:
        logging.info(f"Loaded base weights from {weights_file} and Stage-1 tactile delta from "
                     f"{stage1_checkpoint_dir} (step {stage1_step})")

    # Freeze everything except STAGE2_TRAINABLE_PREFIXES (paper: "train only the latent-to-expert
    # projection and the action expert").
    n_trainable, n_frozen = 0, 0
    for name, p in policy.named_parameters():
        if name.startswith(STAGE2_TRAINABLE_PREFIXES):
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    if is_main:
        logging.info(f"stage2: {n_trainable:,} trainable params, {n_frozen:,} frozen params")

    model = _Stage2Wrapper(policy)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # frozen VL backbone + tactile stack are legitimately unused
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    peak_lr = config.lr_schedule.peak_lr
    warmup_steps = config.lr_schedule.warmup_steps
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_stage2_checkpoint(model, optim, config.checkpoint_dir, device)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Stage2", disable=not is_main)
    infos = []
    epoch = global_step // batches_per_epoch
    while global_step < config.num_train_steps:
        if use_ddp:
            torch_loader.sampler.set_epoch(epoch)
        for observation, actions in itertools.islice(loader, batches_per_epoch):
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32).to(device)  # noqa: PLW2901

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            optim.zero_grad()
            losses = model(observation, actions)
            loss = losses.mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite Stage-2 loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.optimizer.clip_gradient_norm)
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError("Non-finite Stage-2 gradient")
            optim.step()
            global_step += 1

            info = {"stage2_action_mse": float(loss.detach()), "grad_norm": float(grad_norm),
                    "lr": optim.param_groups[0]["lr"]}
            infos.append(info)

            if is_main and (global_step == 1 or global_step % config.log_interval == 0
                            or global_step == config.num_train_steps):
                mean_info = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
                logging.info(f"step={global_step} " + " ".join(f"{k}={v:.4f}" for k, v in mean_info.items()))
                if config.wandb_enabled:
                    import wandb

                    wandb.log(mean_info, step=global_step)
                infos = []
            elif not is_main:
                infos = []

            save_stage2_checkpoint(model, optim, global_step, config, is_main)
            if is_main:
                pbar.update(1)
        epoch += 1

    if is_main:
        pbar.close()
        logging.info("Stage-2 training complete.")


def main() -> None:
    init_logging()
    config = _config.cli()
    try:
        train_loop_stage2(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
