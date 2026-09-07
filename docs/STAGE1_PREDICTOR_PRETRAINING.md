# Stage-1 Predictor Grounding on Human Tactile Data

This implements [N0-VTLA, Section 4.2](https://arxiv.org/pdf/2607.23782), not the Section
4.1 base pretraining pipeline. The upstream release supplies the policy and action-based
training, but not this supervised predictor-grounding trainer. Stage 1 needs a pretrained
base policy and learns the tactile pathway without robot action labels. It does not
reproduce the paper's complete large-scale pretraining.

## Objective and Data

The current branch encodes `tactile(t) - tactile(episode_start)` and predicts latent tokens
using frozen vision-language context. The target is the active-view mean of encodings of
`tactile(t+H) - tactile(t)`, with H=50 in the current config. Mean-pooled, L2-normalized
predictions and targets enter symmetric InfoNCE. A reconstruction head predicts a coarse
future difference field with an L1 loss (Equations 2-5).

Only the tactile projection, predictor, and reconstruction head train. The base policy
stays in eval mode without gradients; the DINOv2 backbone stays frozen. Stage 2 and Stage 3
require actions and are not implemented by this trainer.

Data needs synchronized RGB and tactile videos, episode boundaries, FPS, and a task prompt
(or the configured fallback). The loader samples baseline/current/future tactile frames
and excludes tail-clamped future frames. The baseline must actually represent no contact;
frame zero of an arbitrary human recording does not guarantee this. At 30 FPS the target
is about 1.67 seconds ahead. Timestamp alignment must be verified before training.

The Stage-1 config does not request action sequences or apply action deltas/normalization.
State/action columns need not exist. `Stage1ObservationOnly` supplies zero placeholders
for the inherited interface and ignores real state/action columns. The Pi0.5 tokenizer
**does see a constant zero state in its prefix**: this is a human-data adaptation, not
measured proprioception. Placeholder actions never enter the loss. Robot experiments
needing real state conditioning should use a separately reviewed data path.

## Differences and Unspecified Choices

Existing hyperparameters are retained. This is not an exact reproduction claim.

### ITW pressure-only data (tacWAM V10/V11)

The current adapter uses only normal pressure, never shear. It follows tacWAM
commit `1e4ad399718e8adc80aaf8b0185d17cd0450766f`,
`tacwam/cosmos_tactile/pad_data.py::fit_pad_normalization`:
30 fixed physical-pad parameter sets (left 15, then right 15), fitted on train
recordings only. Baseline is P5 and raw scale is P99.9 minus P5. Scale is floored
at 5% of the median positive raw scale (with the reference's numerical floors).
Pressure is `clip((raw - baseline) / scale, -1, 8)`. Four sampled frames per hand
per recording and seed 42 match the reference defaults. No per-episode or
per-frame rescaling is performed; the same physical pad uses the same parameters
in every episode. Left/right pads do not share a single maximum.

Fit statistics explicitly, using the official recording split, then reuse the
resulting immutable JSON for every conversion. A fit on the downloaded 08/03
training subset is a development version, not the original full-corpus V10 fit.
Do not mix datasets rendered using different statistics; a later refit requires
a new data version and re-rendering. Original tacWAM normalization JSON files
can also be supplied directly when the exact original parameter values are wanted.

```bash
python scripts/itw_pressure.py --raw-root /path/to/itw08-03 \
  --split-manifest /path/to/recording_split_v7_post0802.json \
  --output /path/to/normalization_0803_train_v1.json
python scripts/itw_tactile_smoke_adapter.py /path/to/itw08-03 /path/to/new_dataset \
  --normalization /path/to/normalization_0803_train_v1.json --max-episodes 2
```

MP4 transport is additional to tacWAM's float normalization: R=G=B is
`round((normal + 1) * 255 / 9)`, with black outside the fixed hand-pad layout.
Normalized zero maps to gray 28, not cyan. Quantization is about 0.0353 normalized
units per code value; H.264 CRF18 introduces further loss. Weak contacts can be
attenuated. This is not numerically identical to feeding tacWAM float tensors,
and a faint preview is not grounds for silently boosting each episode.

All three RGB streams and both tactile streams are resampled to the shared
30 Hz overlap using nearest timestamps (ties left), with 17.5 ms camera and
20 ms tactile error gates. The initial converter rejects an entire episode
with bad alignment, unlike tacWAM's window-level filtering. It never deletes
individual ticks and compresses time. Source indices/timestamps and encoding
provenance are saved alongside the canonical metadata. State/action remain
zero interface placeholders for action-free Stage 1, not robot training labels.

| Item | Current implementation | Paper / limitation |
|---|---|---|
| Latent count | 5 | Paper uses 10. Mean-pooled InfoNCE supports either, but the architectures differ. |
| Temperature | 0.07 | Equations 3-4 use unscaled cosine logits, equivalent to temperature 1. Temperature scaling is a deviation, not a requirement for learning. |
| Target gradient | Detached, shared projection | Stop-gradient is our assumption; the paper does not specify it explicitly. Targets change as the shared projection updates through the current branch. No EMA target is used. |
| Reconstruction | View/channel-averaged 8x8 field, two-layer MLP, weight 0.5 | Head details, resolution, aggregation, and weight are not fully specified. Current pressure-only RGB channels are identical, avoiding the old pressure/shear cancellation. |
| Initialization | Load compatible base, including tactile weights if present | Section 4.2 describes a newly initialized tactile pathway. Loading an already-grounded pathway is continuation, not that initialization. |
| LR, batch, duration | Existing defaults unchanged | Not verified as the paper's Stage-1 hyperparameters. |
| EMA | Not implemented | The inherited `ema_decay` field is not applied; checkpoints contain online weights. |
| Sensor | Rasterized human pressure arrays | Paper uses vision-based tactile sensors; rasterization changes the distribution seen by frozen DINOv2. |

Stop-gradient and grayscale reconstruction are retained experimental choices, not claimed
to be required by related work. Validate these assumptions before interpreting losses as
evidence of physical contact prediction.

## Distributed Training and Checkpoints

InfoNCE gathers predictions with autograd and targets/masks across all ranks before
masking invalid rows. Negatives come from the global batch. Reconstruction gradients
use the global valid count, including when one rank has no valid targets. Entirely invalid
batches finish backward on all ranks but skip AdamW, including weight decay. A full epoch
of consecutive empty batches raises an error.

Checkpoints include the full policy and optimizer; disk usage includes the frozen backbone.
Retention pruning is not implemented. Step numbers count completed updates; legacy
zero-based checkpoints resume at the next update. Resume restores weights, optimizer,
and update count, but not exact RNG or the data cursor. The reconstruction head is
Stage-1-only; downstream loaders must tolerate its extra keys. Robot action statistics
must be computed separately before action-based training.

## Run and Verification

On the lab's Blackwell GPUs (`sm_120`), use the CUDA 12.8 wheel of PyTorch 2.7.1.
The CUDA 12.6 wheel imports successfully but fails GPU execution with "no kernel image".
Installing repo dependencies can replace the base image's torch, so verify the runtime:

```bash
python -m pip install --no-cache-dir torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())'
```

Run on lab inside Docker after local commit/push and server pull:

```bash
export VTLA_DATASET_PATH=/path/to/canonical_human_dataset
export VTLA_ASSET_ID=itw_canonical
export VTLA_PRETRAINED_CHECKPOINT=/path/to/compatible_base_checkpoint
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train_stage1_predictor.py vtla_stage1_predictor_pretrain \
    --exp-name=itw_stage1_v0
```

Existing overrides: `VTLA_STAGE1_FUTURE_OFFSET=50`, `VTLA_STAGE1_RECON_GRID=8`,
`VTLA_STAGE1_LAMBDA_REC=0.5`, `VTLA_STAGE1_TEMPERATURE=0.07`. No automatic tuning is done.
Use a new experiment name or `--resume`. Existing run directories are protected unless
overwrite is explicitly requested. Missing/incompatible base checkpoints are fatal.

CPU objective and distributed regression tests are separate from an end-to-end GPU run
with video decoding, tokenization, DINOv2, and PaliGemma. Passing the former does not
establish data quality or convergence of the latter.
