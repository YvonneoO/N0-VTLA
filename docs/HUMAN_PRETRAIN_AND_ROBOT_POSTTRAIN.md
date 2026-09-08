# Human Pretraining and Real-Robot Post-Training

## Scope and Evidence Status

This document stages revisions for the existing Claude artifact, keeping the
robot experiment separate from, rather than replacing, the human-data study:
https://claude.ai/code/artifact/2b3852f7-2e08-4083-a3c6-3428850c38f0

The published artifact was read in the browser. Its current viewer exposes
sharing and version history but no content editor. These revisions are staged
in the fork; they have NOT yet been saved into that remote artifact.

Robot dataset: `SingleBicycle/tacwam-wetlab-tasks`, inspected revision
`ca16780a64856da90033d0270822a806ab3eb53c`. Public metadata lists 53 H5 episodes,
all under `smoke_test/`; the dataset card identifies these as `cap_to_tray`.
Access for `qqyang` has now been granted. One pinned episode was downloaded and
audited in the lab Docker; details and limitations are recorded below. The
reference loader was read, not executed. Statements in the dataset-card section
remain card claims unless explicitly verified in the sample audit.
Credentials are never part of this document or the repository.

## Section A: Human Visual-Tactile Pretraining

### Research Question

Can synchronized human RGB and tactile recordings improve learned tactile
representations and subsequent robot task performance? Distinguish this from
the engineering question of whether the training pipeline runs.

Current implementation is action-free predictor grounding inspired by paper
Section 4.2 Stage 1, warm-started from the released N0-VTLA base, including its
existing tactile weights. It is continued domain adaptation, not from-scratch
base pretraining. The VLA base stays frozen; the tactile projection, predictor,
and auxiliary reconstruction head are updated. No human robot-action labels
are synthesized. See `STAGE1_PREDICTOR_PRETRAINING.md` for exact deviations.

### Verified Progress, Not Yet Dataset-Benefit Evidence

- Fixed pressure-only tacWAM-style normalization: 30 per-hand/per-pad parameter
  sets, fitted on 453 official train recordings and reused across episodes.
- Initial 3-update smoke run passed data loading, gradients, optimizer updates,
  checkpoint saving, and finite-weight checks.
- Current short-run dataset: 50 train episodes / 21,338 aligned frames; 10
  validation episodes / 5,451 aligned frames, with no overlap and no validation
  contribution to normalization. Validation includes multiple dates.
- Evaluation: 994 fixed held-out timestamps; initialized-head MAE 0.1550473
  versus zero-change baseline 0.00148208. Active subset: 102 samples, respective
  MAEs 0.1585377 and 0.00714330. A lower error than a randomly initialized head
  alone is not evidence of useful forecasting.
- The 2,000-update run remains in progress at this audit. At update 950 the
  logged interval means were InfoNCE 0.9754 and reconstruction L1 0.0423.
  These are training metrics, not held-out improvement or robot success.
- The visualization is observed-history prediction of an 8x8 hand-averaged
  future pressure-change field at +50 frames, not action rollout, future RGB
  generation, or high-resolution tactile generation.

### Experiments Needed to Support the Claims

Use identical robot train/validation/test recordings, action conventions,
normalization, post-training budget, and evaluation conditions across arms:

| Arm | Initialization | Robot post-training inputs | Primary comparison |
|---|---|---|---|
| A | Official N0-VTLA base | RGB + robot state, tactile disabled | Robot RGB baseline |
| B | Official N0-VTLA base | Same RGB/state + tactile | B vs A: downstream tactile benefit |
| C | Base continued with paired human RGB+tactile Stage 1 | Same inputs and recipe as B | C vs B: added human-data adaptation benefit |

B vs A must be trained as an actual modality ablation; masking tactile only at
test time is a robustness probe, not an equivalent training comparison.
C vs B does not isolate dataset quality from additional compute. Add
compute-matched human-data controls (for example temporally mismatched tactile
with otherwise fixed recordings/budget, or an alternative matched corpus) and
data-volume curves before attributing improvements specifically to paired-data
quality. A shuffled-tactile control tests alignment value, not all possible
claims about tactile modality or dataset superiority.

To test human-pretraining RGB-only versus RGB+tactile inputs, a separately
specified RGB-only predictor control is required. Simply disabling every
tactile mask in the current trainer makes samples invalid and can skip all
updates. Also, because current Stage 1 changes only tactile-specific modules,
disabling that entire branch later can remove the very adaptation being tested.
No such ablation has been implemented or run yet.

Use repeated seeds and predeclared held-out task/recording conditions. Report
robot success counts and uncertainty, contact failures, and offline physical
action errors. An offline contact-change video illustrates behavior but does
not by itself prove dataset benefit or closed-loop success.

### Corrections to the Existing Artifact

1. Replace the implication that robot post-training must wait for our Stage 1:
   official N0-VTLA base can directly initialize the independent robot branch.
   ITW alone lacks actions, but a separate robot dataset has now been identified.
2. Correct the smoke-run tile `55.9%`: update 1 has 55/64 = 85.94% valid rows;
   updates 2-3 average 58.5/64 = 91.41%. Across all three updates, 172/192 =
   89.58%. Do not convert a sample count into a percentage without division.
3. Replace "target must not drift" with "target is detached within a backward
   pass; its shared projection still changes across updates." No EMA target
   encoder is implemented.
4. Label the two-episode result as the initial smoke run, and add the current
   50/10-episode short-run protocol rather than overwriting historical results.
5. Describe DDP evidence as an actual CPU two-process gloo regression, not a
   verified GPU-DDP training run. GPU-DDP remains unverified on this server.
6. A completed optimizer run produces a checkpoint, not proof that its tactile
   pathway has learned useful grounding. Preserve that distinction in badges.

## Section B: Official Base to Real-World Robot Policy

### Independent Goal

Starting from the official `NeoteAI/n0-vtla-base` checkpoint and fixed
`cap_to_tray` robot demonstrations, determine whether task post-training can
produce a policy that succeeds in supervised real-world deployment. This is
paper Section 4.3 task adaptation, not our human Stage 1, and does not require
its checkpoint. This section corresponds to Arm B above and provides the
robot feasibility baseline for later comparisons.

Success is a measured closed-loop task outcome. A training loss, action replay,
policy server startup, or offline prediction is not real-world success.

### Dataset Card versus Existing Cosmos Experiment

The card describes an xArm6 right arm, Revo2 six-motor right hand, static head
camera, right wrist camera, and a 15-pad / 880-taxel tactile glove. Its 12-D
command contract is xyz (mm), axis-angle (degrees), and six motor targets.
The card recommends last-commanded state and commanded action targets.

In contrast, tacWAM `docs/TRAINING_NOTES.md` records measured robot streams,
15-D xyz + column-based rot6d + six hand values, and backward-framewise action
increments at 10 Hz. Those are choices made by that Cosmos experiment, not the
raw dataset schema or the N0-VTLA post-training contract. The data card's quoted
53-episode statistics and constant-orientation warnings must be checked on the
actual sample; commanded orientation can be constant while measured orientation
still contains sensor noise.

### Code-Verified Compatibility Gaps

| Boundary | Current N0-VTLA behavior | Required robot-data work |
|---|---|---|
| Raw format | `convert_canonical_data.py` reads existing platform CSV profiles | Add a wet-lab H5 adapter, not a path-only substitution |
| Action shape | 32-D container; nine EEF channels + one gripper per arm, then 12 reserved slots | Preserve all six Revo2 channels in an explicit versioned layout; do not average them into a gripper |
| Delta convention | `DeltaActions` subtracts the CURRENT state from every target in the chunk, elementwise on EEF channels | Do not import Cosmos backward-framewise increments or subtract twice |
| Mask semantics | `CanonicalTactileInputs` passes `action_mask`, but `Observation.from_dict` omits it; policy returns unmasked MSE and trainer takes its mean | Metadata masks do NOT suppress invalid targets; preserve zero padding as the original recipe, and explicitly handle temporal validity before training |
| Norm computation | CLI supports Flexiv/ALOHA masks; Flexiv covers only the first arm block | Norm computation, loader deltas, and inverse decoding must use the same proposed layout; the current Flexiv default is wrong for a right-block mapping |
| Timing | Reference horizon 50 at dataset FPS; canonical conversion is 30 Hz | Use a shared 30 Hz timeline, actual source video indices, clutch boundaries, and causal command holds only when justified |
| Tactile input | Fixed image slots with missing-view masks | Use real right glove only; do not treat an unused left stream as measured robot input |
| Serving | Generic websocket/ZMQ policy servers | Add an xArm6/Revo2 observation/command client with matching inverse transforms and safety checks |

A candidate shape-preserving layout is right EEF in `[10:19]`, all six hand
motor values in `[20:26]`, and zero placeholders elsewhere. This keeps the
32-D projection shapes and existing EEF delta mask; it does NOT mean reserved
channels already have pretrained finger semantics. It differs from the shipped
single-arm example's first-ten-slot convention and requires explicit round-trip
tests and matching normalization. This is a proposal, not an implemented or
sample-validated layout.

The full 32-D unmasked action loss is existing behavior. Changing it to a
per-dimension or per-timestep masked loss would change the training objective;
do not silently enable such a change under the request to keep training settings
fixed. Do not fill invalid targets with NaN and expect `action_mask` to hide them.

### One-Episode Audit: Downloaded and Inspected

Episode `094fa583-d87b-5c37-ad38-42d57fa97d48` is stored inside Docker `n0vtla` at
`/DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test/094fa583-d87b-5c37-ad38-42d57fa97d48`.
Selected episode files total 237,702,437 bytes (about 238 MB), excluding the
small root README/loader. No depth, left camera, or other episode was downloaded.
No post-training job or robot movement was started.

Verified observations:

- H5 has 570 rows over 18.967 seconds at approximately 30 Hz. The source trial is
  `cap_smoke_b08_t03_merged`, block `cap_smoke_b08`. `robot/manifest.json` labels
  the trial `success`; `task_info.json.success` is null. Use the trial label,
  not the vendor form or the generic QC verdict, for outcome selection.
- Arm commands are `(570, 6)` xyz mm + axis-angle degrees; hand commands are
  `(570, 6)`. The 244 valid arm-command/clutch rows have finite arm and hand
  targets. Their three commanded orientation dimensions are exactly constant.
  Measured arm and hand streams have only 241 and 242 valid rows respectively
  within that same 244-row subset, despite finite values elsewhere.
- Clutch spans rows 171-480 (310 rows), but only 244 rows satisfy arm validity.
  Those valid rows form 60 disjoint runs, longest 24 frames: there are NO
  contiguous 50-frame all-valid arm-target windows. All 326 invalid arm rows
  still contain finite values, so finiteness alone is not a validity check.
  Inspect raw command timestamps and establish bounded causal hold semantics;
  do not compact valid rows and label the resulting sequence 30 Hz. The
  reference loader does compact rows, so it is not a drop-in temporal-chunk
  loader for this training recipe.
- Both RGB files are H.264, 1280x720, 30 fps. Head contains 7,731 frames;
  right wrist contains 7,736. Three task frames per camera decoded successfully.
  Valid task indices are head 1731-2039 and wrist 1733-2041, confirming that the
  files cover a whole recording block, not just this 19-second trial.
- Cross-host synchronization remains UNVERIFIED. The manifest applies 0 ms
  offset, has no clap marks, and explicitly rejects the motion-correlation
  estimate (-1266.7 ms) as broad/shallow and not confident. Do not apply that
  rejected estimate. The QC `PASS` and timestamp-nearest coverage do not prove
  physical RGB/robot/tactile synchronization. Frame-alignment yaw calibration
  is a spatial transform, not a clock-offset correction.
- Right NPZ contains 7,807 whole-block samples and 15 pressure arrays totaling
  880 taxels; 568 samples fall within the H5 time interval using supplied clocks.
  The H5 tactile arrays are only 15-pad summaries, not full-resolution taxels.
  Its `frame_id_unwrapped` length is also 7,807, not 570: do not index it as an
  episode-row-aligned array. These counts do not resolve the clock uncertainty.
- Pressure is extremely sparse. Over that nominal episode interval, only pads
  3, 7, 13, 15, and 18 have nonzero pressure; the largest raw value is 0.000878119.
  Applying the EXISTING human train normalization with the existing encoder
  erases all variations in pads 3 and 7 (gray 28 only); pads 13, 15, and 18 span
  only gray 28-29. All other pads are gray 28. This loss occurs BEFORE H.264.
  The same normalization METHOD can be retained, but blindly reusing the human
  numerical scales is unsuitable here. First verify units/calibration, then
  fit fixed per-pad robot TRAIN statistics using the same method, shared by all
  robot comparison arms; do not fit per episode or on validation data. No such
  statistics have been changed or fitted during this audit.
- Revo2 fingertip force is a distinct five-channel signal (observed max 2.98 N),
  not a replacement with the same units as the glove taxels. QC reports ring
  status invalid throughout and thumb invalid in 83.1% of raw samples. Preserve
  sensor-specific masks and provenance; do not silently fuse these streams.

The sample establishes available modalities and exposes concrete blockers; it
does not certify the whole dataset or policy readiness. Remaining steps:

1. Actual dataset keys, shapes, timestamps, quaternion order, axis-angle units,
   base/TCP/flange frames, hand motor order and units; compare command and
   measured streams, not just vector dimensions.
2. Successful task label, clutch intervals, separate arm/hand validity, command
   age, nonfinite samples, and camera/tactile coverage.
3. Video frame indices into potentially whole-block videos. Never assume that
   video frame zero equals H5 episode frame zero.
4. Preserve timeline gaps: select valid windows or bounded causal zero-order
   holds with an explicit age rule. Do not concatenate disjoint valid rows into
   a fictitious 30 Hz trajectory; never use future values to fill past commands.
5. Constant dimensions, low-variance channels, rotation conversion and action
   normalization round trips, plus a comparison of decoded actions to source.
6. Fixed tactile scale compatibility with the robot glove, including saturation,
   weak-signal quantization, baseline contact, and missing-pad behavior.

An example adapter design is measured state plus actual issued command targets,
but that is not the card's all-commanded-state contract or Cosmos's measured
future-motion contract. Select and version the intended contract after the
sample audit; make its online source available at inference. Do not silently
substitute one for another.

### Post-Training and Deployment Gates

1. Lock robot train/validation/test IDs and keep whole recording blocks together
   where trials share recordings; inspect time overlap and duplicated media.
   Reuse the exact split in all comparative arms. One audited episode is for
   compatibility checking, not evidence of training or generalization.
2. Convert RGB, tactile, state, actions, and provenance to canonical data.
   Compute state/action statistics on robot TRAIN data only, after the same
   delta transform used by the trainer. Preserve the chosen pressure encoding.
3. Verify shape-compatible official base loading. Run data/inverse-transform
   tests, one complete batch, finite loss/gradients, and checkpoint round-trip.
4. Keep `vtla_tactile_posttrain` learning settings: batch 64, horizon 50,
   500-step warmup, peak LR 2e-5, decay LR 2e-6 at 20,000 steps, clip 1.0,
   tactile-specific LR multiplier 0.1, supervised predictor loss off. A short
   smoke duration must be labeled as such; do not run the human Stage-1 trainer.
5. Before robot motion: physical-unit held-out xyz/rotation/motor errors,
   smoothness, command limits, causal sensor latency, and source-action replay
   through the exact deployed decoder. Offline observed-history action
   predictions are not closed-loop rollouts.
6. Only after those gates, perform supervised low-speed deployment with
   workspace/joint/motor limits, velocity/acceleration caps, stale-observation
   rejection, watchdog and accessible emergency stop. Use receding-horizon
   replanning rather than blindly executing the entire 50-frame chunk.
7. Record attempted and successful trials, failures, interventions, task
   conditions and uncertainty. Define success before testing: cap released
   inside the intended container and hand clear, not merely reaching near it.
   Use the same conditions and execution settings for comparison policies.

No robot adapter, post-training job, or real-world deployment is claimed as
completed by this document. HF access and the first sample inspection are now
complete. Remaining gates include clock alignment, command/state semantics and
causal resampling, tactile units/scales, canonical conversion and inverse tests,
and access to the original artifact editor for publishing these revisions there.
