import dataclasses
import logging
import pathlib
import socket

import tyro

from n0vtla.policies import policy as _policy
from n0vtla.policies import policy_config as _policy_config
from n0vtla.serving import websocket_policy_server
from n0vtla.training import checkpoints as _checkpoints
from n0vtla.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "vtla_tactile_posttrain").
    config: str
    # Checkpoint directory (e.g., "checkpoints/vtla_tactile_posttrain/exp/20000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Checkpoint to serve.
    policy: Checkpoint

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Inference-only memory optimization: construct the PyTorch model's backbone on the meta
    # device and load checkpoint weights via assign=True instead of building a full fp32 CPU
    # copy first. Cuts the load-time system RAM peak roughly in half (measured: ~16.6GB ->
    # ~9.6GB) at no cost to correctness (see docs/REAL_ROBOT_INFERENCE.md §2.2). Off by default;
    # only applies to N0VTLAConfig-based configs (e.g. vtla_tactile_posttrain) and is a no-op for
    # the JAX loading path.
    low_cpu_mem_usage: bool = False

    # Seed torch/numpy/CUDA once at startup, so the sequence of sampled actions is reproducible for a given
    # sequence of requests (each call still draws fresh noise; two calls with the same observation differ).
    # None (default) = unseeded, the previous behaviour.
    seed: int | None = None
    # Multiplier on the Gaussian noise the action chunk is denoised from (PyTorch models only). 1.0 (default) =
    # unchanged. Smaller values shrink the spread between samples; 0 starts from zeros and makes the reply a
    # deterministic function of the observation, but not necessarily a better one: it can bias the chunk toward a
    # blurred average of the modes. For diagnosing whether sampling noise drives rollout variance.
    noise_scale: float = 1.0


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    train_config = _config.get_config(args.policy.config)
    if args.low_cpu_mem_usage:
        if not hasattr(train_config.model, "low_cpu_mem_usage"):
            raise ValueError(
                "--low-cpu-mem-usage is only supported for N0VTLAConfig-based configs, but "
                f"'{args.policy.config}' uses {type(train_config.model).__name__}."
            )
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, low_cpu_mem_usage=True)
        )

    # The config preset's own asset_id (used to locate norm_stats.json under the checkpoint's
    # assets/ dir) is whatever the PRESET defaults to, not necessarily what a given training RUN
    # actually used if it overrode asset_id at launch time (e.g. this repo's own
    # wetlab_v2_train_full_run1 checkpoints were trained with asset_id="wetlab_v2_train", but
    # every vtla_tactile_posttrain preset config defaults to "canonical_tactile_task" -- a mismatch
    # that would otherwise 404 on a totally healthy checkpoint). If the preset's asset_id isn't
    # present under this checkpoint but there's exactly one asset directory, that's unambiguously
    # the checkpoint's own norm_stats -- use it instead of failing.
    norm_stats = None
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    assets_root = pathlib.Path(args.policy.dir) / "assets"
    if data_config.asset_id and not (assets_root / data_config.asset_id).exists() and assets_root.is_dir():
        candidates = [p.name for p in assets_root.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            logging.info(
                "Config asset_id '%s' not found under %s; falling back to this checkpoint's own "
                "asset dir '%s'.",
                data_config.asset_id,
                assets_root,
                candidates[0],
            )
            norm_stats = _checkpoints.load_norm_stats(assets_root, candidates[0])

    policy = _policy_config.create_trained_policy(
        train_config, args.policy.dir, default_prompt=args.default_prompt, norm_stats=norm_stats
    )
    if args.noise_scale != 1.0:
        if args.noise_scale < 0:
            raise ValueError("--noise-scale must be >= 0")
        model = policy._model  # noqa: SLF001
        if not hasattr(model, "sample_noise"):
            raise ValueError("--noise-scale is only supported for PyTorch models")
        original = model.sample_noise
        model.sample_noise = lambda shape, device: args.noise_scale * original(shape, device)
        logging.info("Action noise scaled by %s", args.noise_scale)
    return policy


def main(args: Args) -> None:
    if args.seed is not None:
        import random

        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logging.info("Seeded torch/numpy/CUDA with %d", args.seed)
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        # Only used for the log line below; an unresolvable hostname must not take down a
        # server that has already paid for loading the checkpoint.
        local_ip = "unresolved"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
