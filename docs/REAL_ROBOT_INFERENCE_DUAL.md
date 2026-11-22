# Real-Robot Inference: Dual-Arm Interface Contract (Task 4, tube pour)

Companion to `REAL_ROBOT_INFERENCE.md` (single-arm). Read that document for the server, GPU/RAM
requirements, the websocket client and the safety gates; **this one only lists what changes for
the dual-arm Task 4 checkpoints**. Where the two disagree, this one wins for Task 4. The code that
actually consumes the observation is `n0vtla/policies/canonical_tactile_policy.py`
(`CanonicalTactileInputs`); read it if anything here looks off.

**Status.** Written from the code and the data release, not exercised on a robot. Nobody has run
`serve_policy.py` against a dual-arm checkpoint on live hardware. `scripts/check_dual_policy_replay.py`
runs a checkpoint through the same policy object the server builds, on dataset frames, and checks
the decode (see section 8); run it on the real checkpoint before trusting this document.

## 1. What is different from the single-arm contract

| | single-arm (`REAL_ROBOT_INFERENCE.md`) | Task 4 dual-arm |
|---|---|---|
| RGB views sent | `third_view`, `right_wrist_view` | `third_view`, `left_wrist_view`, `right_wrist_view` |
| Tactile views sent | `right_wrist_right_tactile` | `left_wrist_left_tactile` **and** `right_wrist_right_tactile` |
| State / action size | 32 (right arm + right hand used) | 32, **both arms + both hands used** |
| Active dims | 10-18, 20-25 | 0-8, 10-18, 20-25, 26-31 |
| Tactile normalization file | one hand fitted, copied to both slots | `wetlab_tactile_norm_task4_dual.json`, **each hand fitted separately** |
| Server | `scripts/serve_policy.py` | unchanged (see 7) |

The model is not different: the 32-dim canonical layout already reserved the second arm
(`n0vtla/policies/canonical_schema.py`), and the mid-train data used the same two tactile slots.

## 2. State (`observation.state`, float32[32])

| dims | content | units |
|---|---|---|
| 0-2 | left arm (arm B) EEF xyz, **arm B base frame** | mm |
| 3-8 | left arm rot6d (first two columns of the rotation matrix, flattened column-wise) | - |
| 9 | left gripper, unused | 0 |
| 10-12 | right arm (arm A) EEF xyz, **arm A base frame** | mm |
| 13-18 | right arm rot6d | - |
| 19 | right gripper, unused | 0 |
| 20-25 | **right** Revo2 hand, 6 motor targets | 0-1000 |
| 26-31 | **left** Revo2 hand, 6 motor targets | 0-1000 |

- Like single-arm, this is the **last commanded** target per arm/hand, not encoder readback. Before
  the first command of an arm, use its measured pose (that is what the training data does:
  `n0_dual_adapter.resolve_dual_window` holds an unengaged arm at its last command, or its measured
  start pose before its first command).
- Axis-angle in degrees to rot6d: `n0_dual_adapter.pack_dual` (or `Rotation.from_rotvec(np.deg2rad(aa))`
  then the first two matrix columns, `rot[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)`). Use
  `n0vtla.policies.rotation_utils`; do not hand-roll the convention.
- Each arm's pose is in **its own base frame**. The bases are side by side facing the same way, arm B
  about 600 mm to the left of arm A (580-620 mm, tape-measured, not calibrated;
  `config_snapshot/dual_arm_setup.json`). The model never sees a shared frame.
- Hand motor order is the order of `robot/<side>/hand.jsonl` `target` in the release.

## 3. Images

| key | source | notes |
|---|---|---|
| `observation.image.third_view` | static head RGB camera | not egocentric; restore the camera placement from `camera_references/dual_pour/` first |
| `observation.image.left_wrist_view` | wrist camera on arm B (left) | **new vs single-arm; do not omit** |
| `observation.image.right_wrist_view` | wrist camera on arm A (right) | |

HWC uint8 RGB, any resolution; the pipeline letterboxes to 224x224 (training data was letterboxed the
same way, aspect preserving with black padding). An omitted view becomes a zero placeholder with its
mask false, which the model never saw during Task 4 training. In 6 training trajectories
(`183327` c001-c005 and `184550` c001) the left wrist camera was rotated by about 5.8 degrees and
shifted; it was realigned to the reference afterwards, so align your live camera to the reference.

## 4. Tactile (both hands)

Send each hand as a `(2, H, W, 3)` uint8 stack `[baseline, current]`, key
`observation.image.right_wrist_right_tactile` (robot right hand) and
`observation.image.left_wrist_left_tactile` (robot left hand). The baseline is the **first frame
after `reset()`** for that hand, held fixed (same mechanism as single-arm section 4.2); keep one baseline
per hand. Send a single 3-D frame and the baseline slot is masked, which degrades the contact signal.

Encoding is the single-arm one (`itw_pressure.normalize_pressure` -> `pressure_rgb` -> pad layout in a
224x224 canvas), with these differences:

- Use **`wetlab_tactile_norm_task4_dual.json`** (30 slots: 0-14 = robot LEFT hand, 15-29 = robot RIGHT
  hand), not the single-arm norm file and not the human-corpus stats. It was fitted on the Task 4
  train episodes, one hand at a time, because the right-hand glove responds much more weakly (grip
  step about 0.1-2.2 against 0.4-9.4 on the left). The file must travel with the checkpoint.
- The released raw files are **crossed**: the glove on the robot right hand is in `left_hand_data.npz`,
  the glove on the robot left hand in `right_hand_data.npz`. The model-side mapping is anatomical:
  robot right hand -> `right_wrist_right_tactile` with `hand="right"`, robot left hand ->
  `left_wrist_left_tactile` with `hand="left"` (mirrored pad layout, handled by
  `itw_pressure.rasterize_pressure_frame`). Check your live sensor SDK's channel order directly; the
  crossing is a property of the delivered archive, not necessarily of your hardware.

## 5. Prompt

Leave the default `"Perform the task."`; the checkpoint was trained with it.

## 6. Response (`actions`, float32[50, 32])

Already denormalized and made absolute (`Unnormalize` then `RelRotAbsoluteActions`, which transforms
**both arms** because no `action_mask` is sent at inference). Decode with the same layout as section 2:

```python
import sys; sys.path.insert(0, "scripts")
import n0_dual_adapter as nd
arms, hands = nd.decode_dual(actions)          # {"left"/"right": (50,6)}, {"left"/"right": (50,6)}
# arms[side][:, :3] = xyz mm in that arm's base frame, arms[side][:, 3:] = axis-angle in DEGREES
# hands[side]       = 6 Revo2 motor targets (0-1000) for that hand
```

- Send **each arm its own targets in its own base frame**, and each hand its own motor targets.
  The README of the data release is explicit about this.
- Clip hand targets to 0-1000 before sending; the replay check reports how often the raw prediction
  leaves that range.
- Clutch: in the training data each arm has its own clutch and an unengaged arm holds its last command.
  The model has no clutch input, so engagement/hold logic belongs to your controller.
- Execute the 50-step chunk (1.67 s at 30 Hz) or replan receding-horizon; the checkpoint was not trained
  for a particular cadence (same remark as single-arm section 5).

## 7. Server

`scripts/serve_policy.py` reads the checkpoint's own `assets/<asset_id>/norm_stats.json`
(`task4_dual_pour_train`), built into the checkpoint at training time, and falls back to it if the
config's asset id differs and exactly one asset directory exists. I found no arm-specific code in it or
in `policy_config.py`; the dual-arm handling lives in the transforms and the data, so the server needs
no change. Launch exactly as single-arm:

```bash
python scripts/serve_policy.py --policy.config=vtla_tactile_posttrain \
  --policy.dir=<Task 4 checkpoint dir> --low-cpu-mem-usage
```

Two arms, three cameras and two tactile views mean a longer prefix than single-arm; re-measure latency
and VRAM for this checkpoint instead of reusing the single-arm figures.

## 8. Check before moving anything

```bash
sbatch --export=ALL,CKPT=<ckpt dir>,LABEL=<name> deploy/vision/n0vtla_check_dual_replay_siyuan.sbatch
```

For 24 random training frames it builds a client-style observation (3 RGB, 2 tactile stacks, no action
keys), calls the same policy as the server and compares the returned absolute chunk with the dataset's
chunk: xyz error (mm) and rotation error (deg) per arm, hand error (motor units) per hand, next to
the "hold the current command" baseline, plus checks that predicted rotations are valid and hands stay
in 0-1000. Training frames, one stochastic sample each: it checks the decode chain, it does not measure
generalization. Task 4 has no held-out split (operator decision); real-robot rollout is the evaluation.
Run the safety gates of `REAL_ROBOT_INFERENCE.md` section 6 as before; none of them is implemented in
this repo.
