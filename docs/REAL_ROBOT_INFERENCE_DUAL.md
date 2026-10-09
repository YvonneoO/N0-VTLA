# Real-Robot Inference: Dual-Arm Task 4 (50 mL tube pour) — README for deployment

For whoever wires the real rig (two UFACTORY xArm6, each with a BrainCo Revo2 hand, a static head camera,
a camera on each wrist, a Tujian tactile glove on each robot hand) to the Task 4 checkpoints. It is
self-contained; the single-arm `REAL_ROBOT_INFERENCE.md` is only background (same server, same safety
caveats). What is **not** provided: any robot driver, camera/glove capture, or safety layer — you add those.

## 0. Status: what was verified, and what was not

Verified (VISION cluster, ours checkpoint step 20000):
- **Server + client end to end**: `scripts/serve_policy.py` started with the command in section 3 (no
  asset-id override; it fell back to the checkpoint's own `assets/task4_dual_pour_train` norm stats) and
  `scripts/dual_arm_client_example.py` replayed one released training package through the websocket:
  native-resolution RGB, raw taxel readings encoded live, both arms decoded. 6 queries, mean xyz error
  to the demonstrated chunk 5.7 mm (left arm about 1 mm, right arm 4.5–15.5 mm), steady-state latency
  about 207 ms per call on one H200. One package, six stochastic samples: a plumbing check, not a benchmark.
- **Tactile encoding**: encoding raw taxel readings live reproduces the frames stored in the training videos
  (mean absolute pixel difference 0.03 of 255, from video compression).
- **Decode chain** on 24 random training frames (`scripts/check_dual_policy_replay.py`): predicted rotations
  are valid, hand targets stay inside 0–1000, position/rotation/hand errors are below the "hold the current
  command" baseline for both arms.

Not verified: any run on the real robot or with real sensors; the rot6d/axis-angle convention against your
controller's (only self-consistent round trips were tested); VRAM and latency on your GPU; behaviour under
the real camera/glove clocks. Task 4 has no held-out split (data owner's decision), so no offline
generalization number exists; real-robot rollout is the evaluation. The 50 training trajectories contain
successes only, so no failure recovery was learned.

## 1. What to get

**Code.** `git clone https://github.com/YvonneoO/N0-VTLA`, `main` at or after `93370cb`. The package the
server needs (`n0vtla/`, `n0vtla_client/`, `scripts/serve_policy.py`) is identical to the commit the
checkpoints were trained from (`f046c05`). `pip install -r requirements.txt`.

**Checkpoints** (Hugging Face dataset `qqyang/zihiao_real_test`, folder
`n0vtla_wetlab_posttrain/post_train_task4_dual_vision8gpu/`):

| path | what |
|---|---|
| `ours/<step>/` | post-trained from `n0-vtla-base_plus_stage1_human_14000` (base + full human-data mid-train) |
| `baseline/<step>/` | post-trained from the official `n0-vtla-base` (no mid-train), for comparison |
| `wetlab_tactile_norm_task4_dual.json` | per-hand tactile normalization (also in this repo, `scripts/wetlab_tactile_norm_task4_dual.json`) |

Steps are 2000…20000 every 2000 (final = 20000; a `19999` folder is an extra end-of-run save). Each step folder
has `model.safetensors`, `metadata.pt`, `assets/task4_dual_pour_train/norm_stats.json` (the state/action
normalization, read by the server from the checkpoint) and `optimizer.pt` (not needed for inference). 20000 steps,
batch 64, 50 trajectories (181 episodes after cutting at data gaps), loss about 0.0086 at the end for both runs.

## 2. Hardware

One GPU with 16 GB or more, bf16, no multi-GPU. Memory/latency numbers of the single-arm document do not
carry over (three cameras and two tactile views make a longer prefix); measure on your machine. I measured
about 207 ms per call on an H200 and did not record VRAM.

## 3. Start the server

```bash
python scripts/serve_policy.py --policy.config=vtla_tactile_posttrain \
  --policy.dir=<checkpoint step dir> --low-cpu-mem-usage        # websocket, port 8000
```

`--low-cpu-mem-usage` avoids a ~17 GB host-RAM spike while loading. Do not set `VTLA_ASSET_ID`; the server
finds the checkpoint's single asset directory itself.

Two optional flags for diagnosing rollout variance (default off; PyTorch models):

| flag | effect |
|---|---|
| `--seed N` | seeds torch/numpy/CUDA once at startup: the same sequence of requests gives the same chunks across server restarts; two calls with the same observation still draw different noise |
| `--noise-scale S` | multiplies the Gaussian noise the 50-step chunk is denoised from; `S < 1` narrows the spread between samples, `S = 0` makes the reply a deterministic function of the observation (it can blur between modes, so it is not necessarily better) |

Checked on the ours checkpoint (step 20000) with the example client: with `--noise-scale 0` two replays against the same
server returned identical chunks; two independently started servers with `--seed 0` returned identical results for the same
requests. That only shows the flags work, not that they change task success; compare success counts with and without them
on enough trials (10 trials give a 95% interval of roughly 6%–51% for 2 successes).

## 4. Client

Use `n0vtla_client.websocket_client_policy.WebsocketClientPolicy`. **`scripts/dual_arm_client_example.py` is the
reference**: `DualArmClient.reset()/infer()` build the observation below, `encode_tactile()` does the tactile
encoding, `decode_actions()` decodes the reply. Run it against a server to see the whole loop:

```bash
python scripts/dual_arm_client_example.py --package <released episode dir> \
  --norm scripts/wetlab_tactile_norm_task4_dual.json --port 8000
```

### 4.1 Observation (one `infer()` per call; send **no** `action`/`actions` key)

| key | content |
|---|---|
| `observation.state` | float32[32], section 4.2 |
| `observation.image.third_view` | head camera, HWC uint8 RGB, any resolution |
| `observation.image.left_wrist_view` | wrist camera on the LEFT arm (arm B) |
| `observation.image.right_wrist_view` | wrist camera on the RIGHT arm (arm A) |
| `observation.image.left_wrist_left_tactile` | glove on the robot LEFT hand: `(2, 224, 224, 3)` uint8 `[baseline, current]` |
| `observation.image.right_wrist_right_tactile` | glove on the robot RIGHT hand: same shape |
| `prompt` | optional; default and training value is `"Perform the task."` — keep it |

Images are letterboxed to 224x224 inside the pipeline (training data was letterboxed the same way). An omitted
view becomes a zero placeholder the model never saw in Task 4 training, so send all three RGB views. Restore the
head-camera placement from `camera_references/dual_pour/` of the data release first. In 6 training trajectories
(`183327` c001–c005, `184550` c001) the left wrist camera was rotated ~5.8° and shifted; it was realigned to the
reference afterwards, so align yours to the reference too.

**Tactile baseline.** The model uses `current - baseline`. The baseline is the **first frame after `reset()`**
for that hand, held fixed for the whole rollout (one baseline per hand): start with the hands at rest, call
`reset()`, and keep the first encoded frame as the baseline. A single 3-D frame is accepted but masks the baseline.

**Tactile encoding** (`encode_tactile()`): per pad `(raw - baseline[pad]) / scale[pad]` clipped to [-1, 8],
grayscale `round((x + 1) * 255 / 9)`, 15 pads placed on a 224x224 canvas (`itw_pressure`), left hand mirrored.
Use `wetlab_tactile_norm_task4_dual.json` (slots 0–14 = robot LEFT hand, 15–29 = robot RIGHT hand; fitted per
hand because the right glove responds much more weakly). The released raw files are **crossed**: the glove on the
robot right hand is stored as `left_hand_data.npz`, the one on the robot left hand as `right_hand_data.npz`. The
model-side mapping is anatomical (robot right hand → `right_wrist_right_tactile`, `hand="right"`); check the
channel order of your own sensor SDK directly, the crossing belongs to the delivered archive.

### 4.2 State (`observation.state`, float32[32]) and the layout of the reply

| dims | content | units |
|---|---|---|
| 0–2 | LEFT arm (arm B) EEF xyz, in arm B's base frame | mm |
| 3–8 | LEFT arm rot6d (first two columns of the rotation matrix, flattened column-wise) | – |
| 9 | left gripper, unused | 0 |
| 10–12 | RIGHT arm (arm A) EEF xyz, in arm A's base frame | mm |
| 13–18 | RIGHT arm rot6d | – |
| 19 | right gripper, unused | 0 |
| 20–25 | RIGHT Revo2 hand, 6 motor targets (order of `robot/right/hand.jsonl` `target`) | 0–1000 |
| 26–31 | LEFT Revo2 hand, 6 motor targets | 0–1000 |

- State is the **last commanded** target, not encoder readback. Before an arm's first command use its measured
  pose; an arm that is not currently engaged holds its last command (that is what the training data does).
- Axis-angle (degrees) to rot6d: `n0_dual_adapter.pack_dual`. Use it and `n0vtla.policies.rotation_utils`; do not
  hand-roll the convention.
- Each arm's pose is in its own base frame. The bases sit side by side, arm B about 600 mm to the left of arm A
  (580–620 mm, tape-measured, not calibrated; `config_snapshot/dual_arm_setup.json`). The model has no shared frame.

### 4.3 Reply

`actions`: float32[50, 32], **absolute** physical units (already denormalized; relative-to-state targets restored
for both arms). Decode with the same layout as the state: `decode_actions()` or
`n0_dual_adapter.decode_dual(actions)` → per arm xyz mm and axis-angle degrees (in that arm's own base frame),
per hand 6 motor targets (clip to 0–1000). Send each arm and each hand **its own** targets. 50 steps are 1.67 s
at 30 Hz; execute the chunk or replan receding-horizon, the model was not trained for a specific cadence.
The model has no clutch input; engagement/hold logic belongs to your controller.

## 5. Before moving anything

None of the safety layer exists in this repo. Before the first motion: workspace/joint/motor limits per arm,
velocity and acceleration caps, a collision margin between the two arms and the rack (the data release records
controller collision stops from pressing on the rack, error 31), stale-observation rejection, a watchdog and an
accessible emergency stop; run supervised at low speed first. Replay a recorded episode through the exact deployed
decoder (the example client does this on released data) before commanding the robot.

## 6. Known data caveats (they shape what the policy learned)

- 50 successful trajectories only, no failure recovery. One trajectory (`dual_pour_20261005_183327_c003`) ended in
  a controller collision stop although labelled successful.
- Cross-host clocks (cameras/gloves on Windows, robot on Linux) are NTP-disciplined, not hardware-synchronized.
- Trajectories were cut at short gaps in the command stream (181 episodes); the tactile baseline during training
  was the first frame of each piece, so about 58% of training frames had a baseline that already contained contact,
  while a rollout starts from rest. The effect on behaviour is unmeasured; if grasp-force or "is it gripped" behaviour
  looks unreliable, this is the first suspect.
- The left arm is not commanded for about half of the legal training frames (it holds, then lifts the receiving
  tube first); the right arm is almost always commanded.

## 7. Files

| | |
|---|---|
| server | `scripts/serve_policy.py` |
| reference client | `scripts/dual_arm_client_example.py` |
| decode / layout helpers | `scripts/n0_dual_adapter.py` (`pack_dual`, `decode_dual`), `n0vtla.policies.rotation_utils` |
| tactile norm | `scripts/wetlab_tactile_norm_task4_dual.json` |
| decode-chain check (GPU) | `scripts/check_dual_policy_replay.py`, `deploy/vision/n0vtla_check_dual_replay_siyuan.sbatch` |
| server+client check (GPU) | `deploy/vision/n0vtla_check_dual_server_client_siyuan.sbatch` |
| actual input contract (read this if it disagrees with this document) | `n0vtla/policies/canonical_tactile_policy.py` |
