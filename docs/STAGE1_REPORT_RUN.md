# Stage-1 Short-Training Report

This experiment prepares canonical pressure-only data, evaluates initialization,
trains for 2,000 updates, evaluates the saved checkpoint, and renders offline
future-tactile predictions. It is not an action or closed-loop robot rollout.

## Data and Protocol

- Train: 50 official train recordings from 08/03, 21,338 aligned frames.
- Validation: 10 official validation recordings, 5,451 aligned frames, selected
  from completed downloaded dates because 08/03 alone has only four recordings
  passing the strict alignment gates. The candidate roots are 08/03, 08/06,
  08/07, 08/10, and 08/11, in that order. Unmatched IDs are never used.
- Selection rejected 29 train and 25 validation candidates for alignment issues
  before reaching the requested counts. See each dataset's `meta/selection.json`.
- Fixed tacWAM-style normalization is reused from 453 official 08/03 train
  recordings. None of the validation IDs contributed to those statistics.
- Train and validation UUID sets are disjoint, and normalization JSONs match.
- Validation samples every fifth frame where `t + 50` remains in the episode:
  994 observations. There is no future-tail clamping in this sample set.
- Training settings are unchanged except the requested 2,000-update duration.
  Evaluation uses batch 16 and disables augmentation; this does not change
  training batch 64 or its augmentation behavior.

The normalization fit uses more train recordings than this short run's 50;
this is not validation leakage, but it is part of the experimental protocol.
Validation includes cross-date variation and should not be described as solely
within-date validation. Recording isolation does not establish unseen-person
or unseen-task isolation.

## Outputs

Lab Docker checkout: `/DATA2/qianqian/N0-VTLA`.
Launch there with `bash scripts/run_itw_stage1_report.sh` on an available GPU.
The script protects existing experiment directories; after a partial failure,
inspect outputs rather than blindly deleting or overwriting them.

Data: `/DATA2/qianqian/n0vtla_itw_canonical_smoke/itw0803_report_v1`.

Report root: `/DATA2/qianqian/n0vtla_reports/itw0803_pressure_report_v1_2000`.

- `before/metrics.json`, `before/predictions.npz`: initialized model on validation.
- `training.log`: real training log, not interpolated or smoothed synthetic data.
- `after/metrics.json`, `after/predictions.npz`: checkpoint on identical validation timestamps.
- `comparison/comparison.json`: before/after and zero-change baseline metrics.
- `comparison/contact.mp4`: eight-second clip around the maximum true field-change sample.
- `comparison/ordinary.mp4`: first eight seconds of the first validation recording.
- Matching PNGs: representative frames of both clips.

Checkpoint: `/DATA2/qianqian/N0-VTLA/checkpoints/vtla_stage1_predictor_pretrain/itw0803_pressure_report_v1_2000/2000`.

These are output locations, not assertions that a currently running job has
finished. Confirm checkpoint metadata and successful evaluation before reporting
completion or improvements.

## Interpretation

Report sample-weighted MAE, macro-episode MAE, and the zero-change baseline.
The active subset uses a fixed threshold of mean absolute true change greater
than 1/255 in model image-difference units; it is not a physical contact detector.
Small or zero active sample counts must be disclosed, not silently redefined.
No statistical significance is implied by the overlapping sampled frames.

The visualized prediction is the actual reconstruction head's 8x8 field,
averaged over active hand views. Red means increase, blue means decrease, and
white means zero; all before/after/target panels share fixed limits +/-0.1.
Values outside this display range saturate visually, but metrics use unclipped
predictions. Nearest-neighbor enlargement does not create spatial detail.
Source observations advance five frames at a time; playback at 6 FPS preserves
their physical timeline. The contact clip is selected using target energy,
never using the model's error or improvement.

The run warms from the existing base checkpoint, including tactile weights;
only the reconstruction head is new. The default task prompt and constant zero
state remain unchanged. Successful execution or improvement over a random head
alone is not proof of useful contact forecasting; performance against the
zero-change baseline is essential.
