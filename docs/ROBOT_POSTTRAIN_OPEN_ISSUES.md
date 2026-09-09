# Robot Post-Training: Open Issues and Data Cross-Check

## Purpose and scope

This document records what still needs verification in the wet-lab robot post-training
pipeline (see [Robot Post-Training: Lab Setup](ROBOT_POSTTRAIN_QUICKSTART.md) and
[Wet-Lab Post-Training Smoke](WETLAB_POSTTRAIN_SMOKE.md) for status and reproduction
steps), plus facts established by cross-checking against a second project
(`tacWAM`, local checkout at `@TAMU/tacWAM`, not on any shared server) that independently
post-trains its own backbone (Cosmos3) on the same source dataset,
`SingleBicycle/tacwam-wetlab-tasks`.

**The experiment goal is a backbone/pretrain comparison, not a data-pipeline
convergence.** N0-VTLA and tacWAM post-train two different backbones on the same
53-episode `cap_to_tray` delivery so their results are comparable. That means:

- Facts about the *raw data itself* (clock behavior, mislabeled files, physical
  properties of this specific task) transfer directly and should be checked in
  both pipelines.
- Facts about *how each project chooses to represent state/action/tactile*
  (commanded vs. measured, absolute vs. incremental, native rate, whether the
  tactile encoder is frozen) are pipeline-specific decisions tied to each
  backbone's own pretraining contract. N0-VTLA should keep its own
  release-native conventions rather than copying tacWAM's, but every place the
  two diverge needs a documented reason, not silence — otherwise a later
  reviewer cannot tell a deliberate choice from an oversight.

tacWAM's own docs (`tacWAM/docs/TRAINING_NOTES.md`, `tacWAM/docs/progress.html`,
`tacWAM/docs/wetlab/INFERENCE.md`) are local-only references, not synced to lab or
VISION — treat citations to them as background evidence gathered on this machine,
not as paths that resolve on a server.

## 1. Shared-dataset facts (apply to any backbone, established by tacWAM)

These are properties of the raw `tacwam-wetlab-tasks` delivery itself, not of
tacWAM's modeling choices. They should hold for N0-VTLA's use of the same data.

### 1.1 The 30 Hz grid is a software resampling, not a hardware sync guarantee

`episode_30hz.h5` is built by nearest-neighbor resampling each stream's own local
timestamps onto a shared 30 Hz grid (mechanism: `tacwam/tujian_v2.py:161-178`,
`grid_30hz`/`nearest_indices`). This aligns *indices*, not *physical clocks* — the
head/wrist cameras, tactile glove, and robot controller are still independently
clocked hardware.

On the same rig lineage (xArm6 + Revo2 + Tujian glove), a sibling audit found two
concrete defects that a consumer trusting the H5 masks at face value would miss:

- `valid/arm_right`, `valid/hand_right`, `valid/action_hand_right` all reported
  **100% true** even though a raw-timestamp cross-check found 13/813 arm rows and
  153/813 hand rows outside tolerance (`tacWAM/docs/v7/TELEOP_SMOKE_DATA_AUDIT.md:19`,
  rated P0).
- A **Linux (robot) / Windows (cameras)** cross-host offset needed a manual
  **+66.7 ms** correction with a **±115 ms** resolution band — about 3 frames at
  30 Hz — from a shallow cross-correlation peak (0.670 vs. 0.640 at zero lag)
  (`tacWAM/docs/v7/TELEOP_SMOKE_DATA_AUDIT.md:21`).

**No document on either side confirms these two defects were re-verified as fixed
on the actual 53-episode `tacwam-wetlab-tasks` delivery.** Both tacWAM's wetlab docs
and N0-VTLA's adapter consume the delivered H5 `valid/*` masks and
`time/timestamp_ns` grid without redoing this raw-vs-mask cross-check on the
current batch.

This is independent confirmation that the `physical_sync_verified: false` flag
already recorded in `WETLAB_POSTTRAIN_SMOKE.md` is a real, precedented risk class
on this rig, not overcaution. See [§3.3](#33-p1-physical-cross-host-sync-still-unverified).

### 1.2 Suspected tactile left/right channel mislabeling — needs direct verification

tacWAM measured, across all 53 delivered episodes, that the file named
`right_hand_data.npz` is dead (per-episode max median ≈0.001, std exactly 0.0000)
while `left_hand_data.npz` carries the real signal (median up to ~0.97). The rig
has only one physical glove, worn on the right hand — this is a delivery labeling
bug, not a real two-hand asymmetry (`tacWAM/docs/progress.html:363-364`,
`tacWAM/docs/wetlab/INFERENCE.md:87-93`).

N0-VTLA's one-episode audit read only `right_hand_data.npz` and reported it as the
"real right glove pressure" (`WETLAB_POSTTRAIN_SMOKE.md`), with values that are
consistent with tacWAM's dead-channel numbers rather than contradicting them:
largest raw value **0.000878119**, only 5 of 15 pads nonzero
(`HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md`, "One-Episode Audit" section). A value
that small is exactly what a noise floor around a dead channel looks like, and
`left_hand_data.npz` for this episode was never downloaded, so the hypothesis has
not been tested on N0-VTLA's own copy.

**This is the single highest-priority open item.** See
[§3.1](#31-p0-verify-the-tactile-channel-is-not-swapped).

### 1.3 This task's wrist orientation is physically frozen, and float32 storage hides that fact numerically

For `cap_to_tray`, orientation following is disabled at capture time
(`orientation.enabled: false`); the delivered dataset card measured the raw
axis-angle orientation dims as bit-identical across all 14,356 delivered action
frames (mean `[127.286, 0.009, 127.273]`, distinct-value count `1` per dim in the
public dataset card's own reproduction script). tacWAM independently confirms the
attitude sits on the quaternion singularity (mean `wxyz ≈ [3e-5, 0.707, 4e-5,
0.707]`) with median per-tick change 0.0305° (`tacWAM/docs/progress.html:355-359`,
`tacWAM/docs/wetlab/INFERENCE.md:141-146`).

The dataset card's own worked example shows why this is dangerous rather than
merely uninteresting: computing `std` for these dims in **float64** gives exactly
`0` (loud — produces NaN on divide). Computing it in **float32** — the H5's native
dtype, and the common default — gives a *nonzero* std of `8.4e-3` to `1.5e-2`
purely from rounding error, so `(x - mean) / std` silently lands at full-scale
±1.0 with no NaN, no warning, and no visible symptom in training logs. See
[§3.2](#32-p0-constant-dimension-tolerance-is-likely-too-tight) for whether
N0-VTLA's normalization code is exposed to this.

### 1.4 Smoothness reference numbers (useful as a backbone-agnostic ship gate)

Measured on the demonstrations themselves (not on any policy), inside the clutch
window:

| | at 30 Hz (raw command-stream cadence) | at 10 Hz (tacWAM's model cadence) |
|---|---|---|
| row-to-row \|Δxyz\| mean / p50 / p95 | 3.77 / 2.83 / 10.86 mm (dataset card) | 9.38 / 8.38 / 19.62 mm (`tacWAM/docs/progress.html:449`) |
| direction reversal rate | 7.6% (dataset card) | 5.1% (`tacWAM/docs/progress.html:449`) |

tacWAM also measured that the **measured** robot stream is smoother than the
**commanded** stream on this same data (p95 step 6.62 mm vs. 10.86 mm, reversal
rate 0.6% vs. 7.6% at 30 Hz row cadence, `tacWAM/docs/progress.html:400-412`) — see
[§2](#2-deliberate-n0-vtla-specific-choices-kept-different-from-tacwam) for why
this doesn't automatically mean N0-VTLA should switch.

These numbers are backbone-independent and cheap to reuse: before any real-robot
attempt, compare a trained N0-VTLA checkpoint's predicted step size / reversal
rate on held-out training episodes against this table. tacWAM's own postmortem
found a checkpoint whose predicted step was ~2× demonstration scale with a
40–50% reversal rate was noise-dominated and failed live (cost seven trials);
this check takes minutes on one GPU and doesn't require a robot.

## 2. Deliberate N0-VTLA-specific choices (kept different from tacWAM)

These are not bugs. Recorded here so a future reviewer can see they were
considered, not defaulted into.

| Dimension | N0-VTLA (current) | tacWAM (Cosmos3) | Why they can legitimately differ |
|---|---|---|---|
| State/action source | **Commanded** — last issued arm/hand target (`WETLAB_POSTTRAIN_SMOKE.md` state contract), matching the HF dataset card's recommended contract | **Measured** — `obs/robot/right/*`, explicitly not the teleop command stream (`tacWAM/tacwam/cosmos_tactile/wetlab_policy.py:8-15`) | tacWAM's own smoothness numbers (§1.4) are real evidence *for* switching to measured streams. This is worth revisiting once the P0 items below are resolved, but it is a real design decision with a citable tradeoff (measured mixes in tracking lag/sensor noise; commanded is the contract the released N0-VTLA base was pretrained against) — not something to change silently. |
| Action delta convention | Absolute target chunk, subtract **current state once** from all 50 steps (`compute_canonical_norm.py:158-164`, `apply_delta`) | Per-step **backward-framewise** increment, `T_i⁻¹ @ T_{i+1}` between consecutive measured poses; hand channels stay absolute in every row (`tacwam/cosmos_tactile/wetlab_policy.py:197-209`, `:235-274`) | These are different delta definitions with different statistics — do not mix normalization stats or smoothness numbers across them. N0-VTLA's convention matches the official base checkpoint's existing `DeltaActions` transform; changing it would need matching changes through the pretrained action head. |
| Native rate / horizon | 30 Hz, 50-frame horizon (~1.67 s) | 30 Hz grid strided to 10 Hz, 9-frame window (~0.8 s) | Tied to each backbone's own released config; not a data property. |
| Tactile normalization | **Refit** per-robot P5/P99.9 stats from this dataset, do not reuse human-corpus scale values (`wetlab_smoke_adapter.py:69-92`, `fit_smoke_pressure`) | **Do not refit** — reuse human mid-training corpus stats, because tacWAM's tactile encoder weights are frozen during this post-train and refitting would decouple the input distribution from frozen weights (`tacWAM/docs/TRAINING_NOTES.md:238-240`) | N0-VTLA runs the tactile pathway at a nonzero learning rate (`tactile LR multiplier 0.1` in `ROBOT_POSTTRAIN_QUICKSTART.md`) rather than freezing it, so refitting is the defensible choice here — the two projects differ because their freeze policies differ, not because one is wrong. If N0-VTLA ever freezes the tactile encoder for a variant run, this choice should be revisited. |
| rot6d convention | First two **columns** of the rotation matrix (`wetlab_smoke_adapter.py:55`), decoded via the official `n0vtla.policies.rotation_utils.rot6d_to_matrix` rather than a hand-rolled inverse (`wetlab_smoke_adapter.py:61-66`) | Same convention (columns), but only after a real historical bug used **rows** instead — the round-trip self-test agreed with itself while being wrong relative to the actual convention, and was only caught by an independent downstream comparison (`tacWAM/docs/TRAINING_NOTES.md:168-181`) | N0-VTLA is already on the columns convention and already decodes through the official utility rather than its own inverse, which structurally avoids tacWAM's self-test blind spot. Keep this pattern (decode via the shipped utility, not a hand-rolled inverse) for any future encode/decode change — round-trip agreement alone does not prove the convention is correct, only that encode and decode agree with each other. |

## 3. Open issues, ranked

### 3.1 CONFIRMED and fixed: tactile channel was swapped

Verified directly on 2026-09-09, on the pinned revision
`ca16780a64856da90033d0270822a806ab3eb53c`, episode
`094fa583-d87b-5c37-ad38-42d57fa97d48`:

- `robot/manifest.json` (`sides.left`) declares, for this rig, `"present": false,
  "reason": "no arm or hand installed on this side"`, and lists `tactile_left` as
  `"present": false"` in its per-stream table — i.e. **this episode's own manifest
  says there is no physical left glove at all.** This directly rules out the
  alternative explanation that different episodes/blocks legitimately use
  different arms: the rig is single right-arm/right-hand, full stop, and this is
  first-party metadata from the capture system itself, not an inference from
  file contents.
- Despite that, downloading `left_hand_data.npz` (not previously pulled) and
  comparing it against the already-downloaded `right_hand_data.npz` for the same
  episode shows the "left" file carries the real signal: per-pad max values up
  to 1.62 / 0.89 / 0.71 with 13 of 15 pads nonzero, against `right_hand_data.npz`'s
  largest raw value of 0.000878119 with only 5 of 15 pads nonzero (numbers already
  recorded in `HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md`'s one-episode audit). The two
  pads that read exactly zero in *both* files (pad ids 1 and 5) match, consistent
  with those being genuinely disconnected taxel groups rather than a hand-swap
  artifact.
- This confirms the delivered `{left,right}_hand_data.npz` filenames are swapped
  relative to the physical rig for this dataset, and — since both files were
  pulled directly from the canonical HF revision, not through tacWAM's code —
  the swap is in the **delivered data itself**, not introduced by either
  project's pipeline. It should also be reported back to the data vendor
  (tacWAM's docs already flag this as owed feedback,
  `tacWAM/docs/TRAINING_NOTES.md:369`).

**Fix applied**: `scripts/wetlab_smoke_adapter.py` now reads tactile data from
`left_hand_data.npz` but keeps every `hand="right"` label downstream, since the
signal is physically the right glove — only the delivered filename is wrong. A
runtime guard (`pad_std < 1e-4` check) raises if the chosen file ever looks dead,
so a future HF revision that fixes the labeling won't silently regress this into
reading a genuinely dead file.

**Follow-up needed**: the existing `canonical_smoke_v1` canonical dataset and the
completed 3-step smoke checkpoint on lab
(`/DATA2/qianqian/N0-VTLA/checkpoints/vtla_tactile_posttrain/wetlab_cap_to_tray_smoke_single_v1/2`)
were built against the dead channel and should be regenerated from the fixed
adapter before being reused for anything beyond pipeline-plumbing validation.

**Update 2026-09-09 — corroborated by the vendor's own per-episode `qc_gate.json`.**
Tabulated with `scripts/tabulate_wetlab_qc_gate.py` across all 53 episodes: every
episode's own QC gate independently flags `tactile_npz_right.tip_dead.thumb: WARN`
with the detail *"tip pad 0 (thumb) exact zero for the whole episode — dead sensor
or a finger that never touched"* — 53/53, no exceptions, and "exact zero" rather
than merely low-std. The same gate explicitly `SKIP`s `tactile_npz_left` on all 53
with detail *"file exists but the manifest declares tactile absent — not checked"*
— i.e. the vendor's own tooling is aware the file exists, assumes it's an inert
placeholder because the manifest says there's no left-side hardware, and never
checks its content. That is exactly how a live channel ends up filed under the
"absent" side without anyone noticing: the manifest's per-side hardware
declaration and the tactile-file naming are evidently generated by two different
parts of the vendor's pipeline that disagree with each other, not by one check
that would have caught this itself. Full tabulation:
`/DATA2/qianqian/n0vtla_robot_audit/qc_gate_tabulation_v1.json`.

**Update 2026-09-09 — confirmed on all 53 episodes, not just one.** All 53
episodes were downloaded (§3.4, now done) and swept with
`scripts/audit_wetlab_dataset.py`: every episode's `robot/manifest.json` shows
`sides.right.present=true` / `sides.left.present=false` (uniformly single-arm,
per §1.3's caution against assuming this from one sample), and every episode
classifies as `left_live_right_dead` (relative-dominance comparison, not a bare
absolute threshold — an earlier absolute-threshold version misclassified the two
`cap_smoke_b10` trials as ambiguous because they share a hard-linked source
capture per the dataset card's block/trial structure, so the dead channel's own
noise floor is a per-block constant that isn't identical across blocks). Full
per-episode report: `/DATA2/qianqian/n0vtla_robot_audit/dataset_audit_v1.json`
on lab. The swap direction is uniform across the whole delivered batch.

### 3.2 RESOLVED: constant-dimension tolerance is fine as written

Checked 2026-09-09 on the real 59-episode/16,041-frame train split (§3.4): the
rot6d frozen-orientation dims (13-18) come out with `std==1.0`, `mean==0.0`
exactly — i.e. `_CONSTANT_DIM_TOL = 1e-8` (`compute_canonical_norm.py:32`)
correctly caught them as constant. The theoretical concern below turned out not
to apply to this code:

The dataset card's own float32-std demonstration (`8.4e-3` to `1.5e-2` on the raw
axis-angle degrees) is an artifact of **naive float32 accumulation** — numpy
summing ~14k copies of a ~127-magnitude float32 value loses precision at the
0.1-scale, which is what produces that nonzero std. `compute_canonical_norm.py`'s
`RunningStats` never does this: it upcasts every batch to **float64** before
accumulating (`RunningStats.update`, `batch = np.asarray(batch, dtype=np.float64)`).
Since the underlying rot6d values are computed once (in float64, inside
`pack_commands`) from a bit-identical axis-angle input and only cast to float32 at
write time, every frame's float32-quantized rot6d value is identical bit-for-bit,
so float64 accumulation measures a true, exact zero variance — nowhere near the
`1e-8` boundary in either direction. The dataset card's warning is still correct
advice in general (don't accumulate variance in float32), it just doesn't describe
a bug present in this specific implementation.

Cross-check, as a bonus: the fitted state-dim means for xyz (348.9 / 53.8 / 217.1 mm)
match the dataset card's own independently-reported numbers (348.99 / 52.21 / 216.42 mm)
closely, which is independent evidence the pipeline is measuring the right physical
quantities.

### 3.3 P1: Physical cross-host sync still unverified

Unchanged from `WETLAB_POSTTRAIN_SMOKE.md`, now with a concrete precedent (§1.1)
that the failure mode is real on this rig lineage and that H5-reported masks can
be wrong with no visible symptom. Options, roughly in order of cost: (a) ask the
dataset provider directly whether the mask-correctness and clock-offset issues
found on `tacwam-teleop-smoke` were fixed before `tacwam-wetlab-tasks` was
captured; (b) redo the raw-timestamp-vs-H5-mask cross-check
(`tacWAM/docs/v7/TELEOP_SMOKE_DATA_AUDIT.md` §method) on a few of our own
episodes; (c) continue training under the existing `--allow-unverified-sync-smoke`
gate and treat any resulting policy as smoke-only until resolved, as already
documented.

**Update 2026-09-09 — both available verification paths checked out, neither works,
on this delivery specifically:**

- **Clap markers (option (a)'s answer, found without having to ask the vendor):**
  the QC-gate tabulation (§3.1 update) shows `clap_marks: WARN` on 33/53 episodes
  with detail *"fewer than 2 clap marks — cross-host drift cannot be clap-verified
  (record.anchor_clap has no caller yet)"*. That is the vendor's own tooling saying
  the clap-marking feature was never wired up (`anchor_clap` has no caller) for
  this recording session, not that individual claps were missed. There is no
  reason to expect the other 20 "PASS" episodes have a real clap-verified offset
  either, rather than happening to clear whatever minimum the check applies before
  reaching the same code path — this needs the raw per-check numbers to confirm
  either way, not just the verdict string. **Conclusion: clap-based verification is
  not available for this delivery**, full stop; it was never captured, not merely
  unverified after the fact.
- **Optical-flow-vs-TCP-speed motion proxy (option (b), extended to 3 episodes):**
  reran `verify_wetlab_smoke.py motion` on two more episodes
  (`8f21c4f2`, `c817f622`, from different blocks) alongside the original
  (`094fa583`). All three show the *same qualitative pattern*: `first_half` best
  lag ≈ **+2233 to +2400 ms**, `second_half` and `whole` best lag ≈ **-1267 to
  -1767 ms** — consistent in sign and rough magnitude across three independent
  episodes, which rules out this being one episode's fluke. But a lag estimate
  that flips sign between the first and second half of the *same* episode, three
  times in a row, is evidence the optical-flow-vs-speed correlation itself has a
  systematic bias (plausibly: the visual-flow-to-speed relationship differs
  between the reach phase and the retreat phase of a grasp, which this proxy
  cannot distinguish from a clock offset) — not evidence of a stable, extractable
  clock offset. Zero-lag correlation is also below the method's own
  "sharp"/"plausible" bar in every case (`094fa583` audit.json already records
  `sharp: false, plausible: false, reject_reason: "correlation peak too shallow or
  too wide to be a measurement"`).

**Both independent verification paths available on this delivery are unusable**:
clap marks were never captured, and the motion-proxy fallback produces a
reproducible-but-uninterpretable pattern rather than a trustworthy offset.
Getting a real answer needs the vendor to add clap markers (or an equivalent
hardware trigger) to a *future* capture — it cannot be recovered retroactively
from what was delivered. Continue treating any training on this data as
`physical_sync_verified: false`, per the existing gate.

### 3.4 DONE: all 53 episodes downloaded

Downloaded 2026-09-09 to `/DATA2/qianqian/n0vtla_robot_audit/cap_to_tray/smoke_test`
(977 files, ~13 GB — no depth video, both hand npz files now included per §3.1).
Both the constant-dimension check (§3.2) and a real (non-smoke) tactile/pose
normalization fit can now run over the full set or a proper train split of it.

### 3.5 P2: Checkpoint final-step save off-by-one

Unchanged from `WETLAB_POSTTRAIN_SMOKE.md`: `global_step` is incremented before
the final-save comparison against `num_train_steps - 1`, so the last update of a
run is never persisted. Cheap to fix, not urgent while still smoke-testing.

### 3.6 P2: Multi-GPU NCCL initialization stall

Unchanged from `WETLAB_POSTTRAIN_SMOKE.md`. Single-GPU already covers the
recipe's global batch size (64), so this blocks scaling to faster/multi-node
training but not the next round of correctness fixes above.

### 3.7 RESOLVED: an over-strict causal camera-gap check silently discarded 31/53 episodes

Found and fixed 2026-09-09 while batch-converting the full dataset. Before the
fix, `wetlab_smoke_adapter.py` required `gap >= 0` (the mapped video frame at or
before the 30 Hz grid tick) for both cameras. A first full-batch run produced only
22 of 53 episodes; a per-episode diagnostic breaking out every validity mask
separately showed all 53 had a healthy 200-440-row contiguous window on every
*other* condition (arm/hand/tactile age, clutch, engaged window) — the camera
check alone was zeroing out 31 of them.

Root cause: the delivered `src_idx` is a **nearest-frame** mapping, not a
most-recent-past one, so `gap = t - frame_timestamp` straddles zero by design
(typically ±15-30 ms, roughly one 30 Hz frame period) depending on the per-episode
phase between the video and robot clocks. The one episode this adapter was
originally tuned against (`094fa583`) happened to have a favorable phase where gap
stayed positive throughout — that was luck, not a property of the delivery.

Fix: replaced the one-sided `gap >= 0 & gap <= 50ms` cutoff with a **symmetric**
`|gap| <= 17.5ms` tolerance, reusing the exact `CAMERA_ALIGNMENT_TOLERANCE_NS`
constant already established elsewhere in this codebase for the same kind of
camera-alignment check, rather than inventing a new threshold. Note this trades a
small amount of documented "strict causality" (a training window can now include a
camera frame up to 17.5 ms in the future relative to its nominal tick) for a large
reduction in discarded data — worth being explicit about if this dataset is ever
used to justify a causality-dependent claim; it does not affect the tactile/command
causal-hold logic (`causal_indices`), which is unchanged and still one-sided.

Result after the fix: **0 of 53 episodes skipped** (up from 31), producing a
68-episode canonical dataset (some source episodes now split into more than one
qualifying window) with 17,861 total frames — more than the delivered dataset
card's own reported 14,356 action frames, consistent with this adapter's
zero-order-hold construction keeping small command-loop gaps rather than dropping
them (see §2's action-source design-choice row).

## 4. Recommended order of work

1. ~~§3.1 — verify the tactile swap.~~ DONE, confirmed on all 53 episodes.
2. ~~§3.4 — download the remaining 52 episodes.~~ DONE.
3. ~~§3.7 — fix the over-strict camera-gap check.~~ DONE, 0/53 episodes skipped
   (was 31/53).
4. ~~Fit a shared train-split tactile normalization~~ (`fit_wetlab_tactile_norm.py`)
   ~~and build the full canonical dataset~~ (`build_wetlab_canonical_dataset.py`,
   `--workers N` for parallel per-episode conversion) — DONE:
   `/DATA2/qianqian/n0vtla_robot_audit/canonical_wetlab_v1`, 68 episodes / 17,861
   frames (59 train / 9 val), split tagged per episode using
   `wetlab_split_v1_tacwam_match.json` (tacWAM's exact 48/5 episode-random split,
   for direct comparability — see §2's split-choice caveat about leakage risk and
   why checkpoint selection should NOT use this split, only the block-holdout one).
5. ~~§3.2 — rerun `compute_canonical_norm.py --train-only`.~~ DONE and RESOLVED:
   frozen-orientation dims correctly get identity stats; state-dim means
   cross-validate against the dataset card's own numbers. Stats at
   `/DATA2/qianqian/n0vtla_robot_audit/assets/vtla_tactile_posttrain/wetlab_full_v1/norm_stats.json`.
6. ~~§5.1 item 4 — multi-episode sync diagnostic.~~ DONE: reproducible-but-
   uninterpretable pattern across 3 episodes, see §3.3 update.
   ~~§5.1 item 7 — QC-gate tabulation.~~ DONE, see §3.1/§3.3 updates. §5.1.8
   (block-level split) was already done as part of step 4 above
   (`block_holdout_v1` in the split manifest). §5.1 item 3 (raw-vs-mask
   cross-check) is still open, lower priority now that the QC-gate tabulation
   surfaced the more actionable clap-marks/tip-dead findings by itself.
7. **Next**: §5.2 visualizations, especially the per-episode overlay video
   (cheapest way to confirm the tactile fix is also *aligned*, not just *live* —
   see the methodology note in §5.2 about why the encoded video comparison only
   became a meaningful test once normalization was fit once across train rather
   than per-episode).
8. Rerun `verify_wetlab_smoke.py batch` and `checkpoint` checks against
   `canonical_wetlab_v1` and the new `norm_stats.json` (the existing checks were
   written against the single-episode smoke dataset and may need small path/shape
   adjustments for the multi-episode one).
9. Once 6–8 are settled, write down a final decision (with citation to §1.4's
   numbers) on whether to keep the commanded/absolute-delta contract or move to
   tacWAM's measured/framewise-delta contract — this determines what "the same
   data" means for the backbone comparison, so it should be fixed before either
   side's numbers are treated as final.
10. §3.5 and §3.6 (checkpoint off-by-one, multi-GPU) whenever convenient before a
    full-scale run; neither blocks the correctness work above.

## 5. Verification and visualization checklist before scaling past one episode

Ordered by priority. "All episodes" means the full 53-episode `smoke_test/`
delivery once downloaded (§3.4), not just the one audited so far.

### 5.1 Data integrity (must run before trusting any multi-episode fit)

1. **DONE — manifest sanity sweep.** `scripts/audit_wetlab_dataset.py` parses
   `robot/manifest.json` for every episode. Result on all 53: `sides.right.present
   == true`, `sides.left.present == false`, `task.name == "cap_to_tray"`
   uniformly — no mixed-rig or wrong-task episodes found.
2. **DONE — tactile liveness, both files, every episode.** Same script computes
   per-pad std for both `left_hand_data.npz` and `right_hand_data.npz` and
   classifies by relative dominance (not a bare absolute threshold — see the
   `cap_smoke_b10` note in §3.1's update). Result: all 53/53 episodes classify
   as `left_live_right_dead`. The swap direction is uniform across the whole
   delivered batch; report at
   `/DATA2/qianqian/n0vtla_robot_audit/dataset_audit_v1.json` on lab.
3. **Command-vs-mask cross-check on a sample.** tacWAM's sibling
   `tacwam-teleop-smoke` audit found the delivered H5 `valid/*` masks reporting
   100% valid while a raw-timestamp cross-check found real gaps
   (`tacWAM/docs/v7/TELEOP_SMOKE_DATA_AUDIT.md:19`). `wetlab_smoke_adapter.py`
   already re-derives its own causal masks from raw JSONL rather than trusting
   `valid/action_arm_right` blindly, but it does trust `action/right/clutch` and
   the per-camera `valid/{name}` fields straight from the H5. Spot-check a
   handful of episodes' raw command/video timestamps against those H5 fields
   the way the sibling audit did, rather than assuming this delivery is exempt
   from the same defect class.
4. **DONE — cross-host sync diagnostic, three episodes.** `verify_wetlab_smoke.py
   motion` reran on two more episodes beyond the original. Result: reproducible
   sign-flip pattern (first-half lag ≈ +2300ms, second-half/whole ≈ -1300 to
   -1767ms) across all three, which is evidence the motion-proxy method itself
   has a systematic bias, not a usable offset. See §3.3's 2026-09-09 update for
   full numbers and conclusion.
5. **DONE — constant-dimension audit on the real fit.** Ran
   `compute_canonical_norm.py --train-only` on the real 59-episode/16,041-frame
   train split; frozen-orientation dims correctly land in the identity bucket.
   See §3.2 (RESOLVED).
6. **Round-trip and batch checks on the real dataset.** Still open: rerun
   `verify_wetlab_smoke.py batch` (and `checkpoint`, once a real run exists)
   against `canonical_wetlab_v1`, not just the one-episode smoke.
7. **DONE — QC-gate tabulation.** `scripts/tabulate_wetlab_qc_gate.py` across all
   53: all verdicts `PASS`, but with informative non-pass detail underneath — see
   §3.1's 2026-09-09 update (tactile dead-tip corroboration) and §3.3's (clap
   marks never wired up). Full tabulation:
   `/DATA2/qianqian/n0vtla_robot_audit/qc_gate_tabulation_v1.json`. Have not yet
   cross-referenced `ledger.csv` or the `task_info.json.success == null` vs.
   `manifest.trial.label == "success"` discrepancy specifically.
8. **Split by block, not by episode.** Blocks (`cap_smoke_b05`…`b11`) group
   repeated trials from the same session/prop placement. Hold out whole blocks
   for validation/test, per the existing recommendation in
   `HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md` §"Post-Training and Deployment
   Gates" — splitting by episode risks near-duplicate trials leaking across
   train/val.

### 5.2 Visualization (cheap, and catches what statistics alone miss)

1. **Per-episode overlay video** — head cam + wrist cam + tactile pressure
   heatmap (now reading the corrected live pad file) + action trace, in the
   spirit of tacWAM's own `viz/<episode>.mp4`. Watching a handful of episodes to
   confirm pressure visibly rises exactly when the gripper contacts the cap is
   the single cheapest end-to-end confirmation that the tactile fix (§3.1) and
   the time alignment are both doing something physically sensible — statistics
   alone (e.g. "std increased 650x") establish that the channel is alive, not
   that it's aligned to the right moment.
2. **Action trajectory plots** — xyz / rot6d / hand-motor channels over time
   with the engaged/clutch window shaded, across a sample of episodes (not just
   the one audited), to confirm the selected window excludes lead-in/tail per
   the dataset card's §3 (operator resets the prop, retreats) rather than only
   trusting that boundary on one episode.
3. **Per-dimension histograms** across all delivered action frames (raw 12-D
   and canonical 32-D) — reproduce the dataset card's own "distinct values per
   dimension" check on our own downloaded copy, per its explicit "verify this
   yourself" instruction, rather than citing the card's numbers as if measured
   on our data.
4. **Tactile dynamic-range histogram**, post-fix — peak per-frame pressure
   reading across all episodes, since N0-VTLA's normalization philosophy
   (refit robot-only stats, not reuse human stats — see §2) means tacWAM's
   "0.13% of range used" number does not transfer; this needs its own
   measurement under N0-VTLA's own normalization.
5. **Sync-diagnostic overlay plot** — the lag-vs-correlation curve from
   `verify_wetlab_smoke.py motion`, plotted for several episodes on one axis, to
   visually judge whether the lag estimate is consistently unstable or only
   unstable on the one episode audited so far.
6. **Post-fit normalization sanity plot** — once real train-split stats exist,
   a box/violin plot of normalized action values per dimension, specifically to
   catch a recurrence of the constant-dimension-noise failure (§3.2) visually
   before spending a training budget on it.

### 5.3 Only after 5.1-5.2 pass

Rebuild the canonical dataset from all verified-good episodes with the fixed
adapter, refit normalization on the real train split, rerun the batch/checkpoint
checks against it, and only then launch a real (not 3-step) training run.

## References

| Topic | File |
|---|---|
| Current lab setup / status | [ROBOT_POSTTRAIN_QUICKSTART.md](ROBOT_POSTTRAIN_QUICKSTART.md) |
| Smoke run results and alignment evidence | [WETLAB_POSTTRAIN_SMOKE.md](WETLAB_POSTTRAIN_SMOKE.md) |
| Full study framing (human Stage 1 vs. robot post-train) | [HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md](HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md) |
| Adapter implementation | `scripts/wetlab_smoke_adapter.py` |
| Normalization stats implementation | `scripts/compute_canonical_norm.py` |
| Batch/checkpoint verification implementation | `scripts/verify_wetlab_smoke.py` |
| tacWAM wetlab data contract (local-only, not on servers) | `@TAMU/tacWAM/docs/TRAINING_NOTES.md`, `@TAMU/tacWAM/docs/progress.html` (lines 336-627), `@TAMU/tacWAM/docs/wetlab/INFERENCE.md` |
| tacWAM adapter implementation (local-only) | `@TAMU/tacWAM/tacwam/cosmos_tactile/wetlab_policy.py`, `pad_data.py` |
| Same-rig-lineage sync/mask defect precedent (local-only) | `@TAMU/tacWAM/docs/v7/TELEOP_SMOKE_DATA_AUDIT.md` |
