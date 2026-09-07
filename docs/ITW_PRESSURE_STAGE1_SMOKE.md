# ITW Pressure-Only Stage-1 Smoke Run

## Data Version

Run date: 2026-09-07. Implementation: `a88d4e9bac2cc726009b16e684c940960514eae8`.
All preparation and training execute in lab Docker container `n0vtla`.

- Raw date: `/DATA2/qianqian/n0vtla_itw_raw/itw08-03`.
- Split: `/DATA2/qianqian/n0vtla_data_manifests/tacwam_v10_split.json`.
- Fixed normalization: `/DATA2/qianqian/n0vtla_data_manifests/normalization_0803_train_v1.json`.
- Normalization SHA256: `ed9c91636d8f22c8d976bffa5516477b904798dab0df08a2fca1f5d021d78e6f`.
- Canonical data: `/DATA2/qianqian/n0vtla_itw_canonical_smoke/itw08-03_pressure_tacwam_v1`.

Normalization follows tacWAM V10/V11's algorithm, fitted on 453 local train
recordings, four frames per hand per recording, seed 42. There are 30 fixed
per-hand/per-pad baseline/scale pairs, with scales 0.00858294 to 0.971982.
These are development-subset statistics, not the original full-corpus parameters.

The two rendered episodes are both in the official train split:

| Source UUID | Aligned frames |
|---|---:|
| `02452e46-f019-4bda-84e3-86f54ca47f44` | 448 |
| `02947400-4456-4f0a-a528-c57e0e0378b3` | 650 |

Each episode has three RGB and two pressure-only hand videos, 224x224 at 30 Hz.
All 5,490 video frames passed full decoding. Video files total 4,563,298 bytes.
Camera/tactile alignment passes the 17.5/20 ms gates. Metadata contains source
indices, timestamps, encoding, and normalization provenance. No validation or
test episode is used in the smoke run.

## Launch

From the Docker checkout `/DATA2/qianqian/N0-VTLA`:

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu \
VTLA_DATASET_PATH=/DATA2/qianqian/n0vtla_itw_canonical_smoke/itw08-03_pressure_tacwam_v1 \
VTLA_ASSET_ID=itw08-03_pressure_tacwam_v1 \
VTLA_PRETRAINED_CHECKPOINT=/DATA2/qianqian/N0-VTLA/checkpoints/n0-vtla-base \
python scripts/train_stage1_predictor.py vtla_stage1_predictor_pretrain \
  --exp-name=itw0803_pressure_tacwam_v1_smoke_20260907 --num-train-steps=3
```

Use a new experiment name to repeat this check; do not overwrite the existing run.
The only training-duration override is three completed updates. Global batch 64,
eight loader workers, optimizer, LR schedule, horizon, losses, and architecture
remain unchanged. This is single-GPU verification, not a GPU-DDP validation.

Log: `/DATA2/qianqian/n0vtla_logs/stage1_itw0803_pressure_tacwam_v1_smoke_20260907.log`.

## Verified Result

The process exited successfully after three completed optimizer updates.
Model initialization took about 55 seconds; the update loop and checkpoint save
took about 72 seconds. The unchanged configuration trains 123,818,048 parameters
and freezes 3,705,436,177 parameters.

| Logging interval | InfoNCE | Reconstruction | Total | Valid samples / 64 | Gradient norm |
|---|---:|---:|---:|---:|---:|
| Update 1 | 4.0521 | 0.1447 | 4.1245 | 55 | 36.4582 |
| Updates 2-3 mean | 4.0253 | 0.1446 | 4.0976 | 58.5 | 25.7655 |

The final log averages updates 2-3, not update 3 alone. Three updates do not
establish a meaningful convergence trend. Printed LR rounds to `0.0000` because
the logger uses four decimal places; the saved schedule retains 500 warmup
steps, peak LR 1e-4, decay steps 20,000, and final LR 1e-5.

Checkpoint directory:
`/DATA2/qianqian/N0-VTLA/checkpoints/vtla_stage1_predictor_pretrain/itw0803_pressure_tacwam_v1_smoke_20260907/3`.

Saved files total 9,313,676,066 bytes (about 8.67 GiB), including frozen weights
and optimizer state. Metadata records `global_step=3` and
`step_format=completed_updates`. All 38 trainable tensors passed a finiteness
check. Sample projection and predictor weights differ from initialization by
up to 1.20e-6; a sampled frozen action projection is unchanged. This is not a
full checkpoint-resume or all-frozen-parameter comparison test.

The loader emits the existing read-only NumPy-to-tensor warning. It did not
prevent execution; the warning and MP4 quantization limitations remain known
risks rather than evidence of data quality.

## Scope and Remaining Risks

This is the current action-free Stage-1 predictor-grounding implementation
(paper Section 4.2), not full base pretraining (Section 4.1) or robot post-training.
It loads the existing base checkpoint, including available tactile pathway
weights. The reconstruction head initializes randomly. State/actions do not
enter the objective; the tokenizer sees a constant zero state. The unchanged
configuration uses the default prompt, not per-episode task text.

Pressure normalization is already baked into the videos. A loader message about
missing action/state normalization statistics does not mean tactile statistics
were skipped. See `STAGE1_PREDICTOR_PRETRAINING.md` for the precise algorithm and
known paper deviations.

Passing a short smoke run establishes execution, not convergence or contact
prediction quality. Weak pressure is affected by 8-bit quantization and lossy
video encoding. Episode frame zero is not guaranteed to be contact-free. Before
scaling data, enforce the train split during episode selection and retain the
same normalization version across train/validation/test. This run does not
convert all downloaded dates or start the default 20,000-update training job.
