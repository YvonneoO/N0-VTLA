# Real-Robot Inference: Interface Contract

For whoever is wiring up the real xArm6 + BrainCo Revo2 rig to the trained
`wetlab_v2_train_full_run1` checkpoint. This document specifies exactly what the
inference server expects to receive and what it returns — verified by reading
the actual transform code (`n0vtla/policies/canonical_tactile_policy.py`), not
inferred from training-side documentation. **What is NOT provided**: any
robot-specific driver (xArm6 SDK calls, Revo2 motor commands, camera/tactile
capture), or a safety layer (limits, watchdog, e-stop). That is what you are
adding. See §6 before sending a single command to the real arm.

## 0. What's already verified working, and what isn't

- The training pipeline, the checkpoint, and the offline ship-gate check
  (predicted-trajectory smoothness vs. real demonstrations) are all verified —
  see `ROBOT_POSTTRAIN_OPEN_ISSUES.md` and `ROBOT_POSTTRAIN_QUICKSTART.md`.
- **Nobody has run this server against a live client or a real robot.** This
  document specifies the contract from the code; it has not been exercised
  end-to-end with real hardware. Treat the first real connection as a new
  integration test, not a formality.
- **Physical cross-host sync (camera/tactile/robot clock alignment) is
  permanently unverified for the training data** (`ROBOT_POSTTRAIN_OPEN_ISSUES.md`
  §3.3) — a checkpoint clearing the offline ship-gate says nothing about this.
  It does not block live inference (which has its own, real-time clock, not the
  training data's), but it does mean you should not expect the offline
  ship-gate result to predict real-world success.

## 1. Getting the code, and where the data lives

**Code**: this is not duplicated anywhere else — clone the full repo and check
out the exact commit this document and the uploaded checkpoints were built
against, rather than requesting or assembling a partial copy:

```bash
git clone https://github.com/YvonneoO/N0-VTLA
cd N0-VTLA
git checkout 4e60f0c2a08b20aefdf98ae7b3177c6469d8a6d6
pip install -r requirements.txt   # or: see pyproject.toml
```

`serve_policy.py` needs the full `n0vtla/` package (model implementation,
training config, transforms, model loader) and `n0vtla/serving/` (the
websocket server) — all of it is in this repo already; there is nothing
serve-specific missing from a normal clone. An earlier version of this
document's Hugging Face upload included a hand-picked subset of files that
turned out to be missing transitive dependencies (no model implementation, no
training config, no websocket server) — that subset has been removed from
Hugging Face. Don't reconstruct the code from a file listing there; clone the
repo.

**Data** (not in git, genuinely only on Hugging Face / lab): the four
checkpoints and `wetlab_tactile_norm_v2.json` (§4.2) at
[`qqyang/zihiao_real_test/n0vtla_wetlab_posttrain/`](https://huggingface.co/datasets/qqyang/zihiao_real_test/tree/main/n0vtla_wetlab_posttrain)
— `checkpoint_5000/10000/15000/20000/`, `wetlab_tactile_norm_v2.json`,
`ship_gate_15000.json`, `ship_gate_20000.json`. Also present on lab under
`/DATA2/qianqian/N0-VTLA/checkpoints/vtla_tactile_posttrain/wetlab_v2_train_full_run1/`
and `/DATA2/qianqian/n0vtla_robot_audit/wetlab_tactile_norm_v2.json` if you're
working from there directly instead.

## 2. Starting the server

```bash
cd N0-VTLA
python scripts/serve_policy.py \
  --policy.config=vtla_tactile_posttrain \
  --policy.dir=<path to checkpoint_20000, from lab or downloaded from HF>
```

This starts a websocket server (`n0vtla/serving/websocket_policy_server.py`,
default port 8000) that loads the checkpoint once and serves inference
requests. It reads `norm_stats.json` from the checkpoint's own
`assets/wetlab_v2_train/` directory (not the repo's `assets/` dir, and not
`wetlab_tactile_norm_v2.json` — see §4.2 for why those are two different
files), so the checkpoint directory must be complete (`model.safetensors`,
`metadata.pt`, `assets/`).

Run `python scripts/gate_c_check.py` first if you've changed anything about the
model assembly or tactile path (DEPLOY.md) — a compatibility regression test
against a known-good state.

## 3. Client connection

Use `n0vtla_client.websocket_client_policy.WebsocketClientPolicy`:

```python
from n0vtla_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="<server-ip>", port=8000)
policy.reset()              # call once per episode/rollout attempt -- see §4.2 (tactile baseline)
result = policy.infer(obs)  # obs: dict, see §4. result: dict, see §5.
```

`infer()` blocks on one round trip (msgpack over websocket) and returns the
full response dict. There is no batching — one call per observation.

## 4. Observation dict (what you send)

Verified against `CanonicalTactileInputs.__call__`
(`n0vtla/policies/canonical_tactile_policy.py:84-189`), the transform that
actually consumes this dict — `create_trained_policy` does **not** run any
repack step by default, so these are the literal keys the model-facing
pipeline expects, not the raw canonical LeRobot column names of the training
data (they happen to be nearly the same, which is intentional but not
guaranteed by any other layer).

| Key | Required | Shape / dtype | Notes |
|---|---|---|---|
| `observation.state` | **yes** (raises `ValueError` if absent) | `float32[32]` | See §4.1 for the layout. This is the **last commanded** arm+hand target, not measured proprioception — matches the training contract (`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §6). |
| `observation.image.third_view` | recommended | `HWC uint8` RGB, any resolution | Static head camera. Resized to 224×224 internally (`ResizeImages`, `n0vtla/transforms.py:343`). Absent → zero placeholder, `image_mask` false for that view (the model was trained with this view always present for wetlab, so don't omit it in practice). |
| `observation.image.right_wrist_view` | recommended | `HWC uint8` RGB, any resolution | Wrist (D405) camera. Same resize/placeholder behavior. |
| `observation.image.left_wrist_view` | omit | — | This rig has no left arm; leave absent (the model was trained with this view masked absent for every wetlab episode — see `ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.1 for why the rig is right-only). |
| `observation.image.right_wrist_right_tactile` | recommended | `float32[2,H,W,3]` or `uint8[2,H,W,3]` stack `[baseline, current]` (or a single `HWC` frame — see §4.2) | Tactile pressure heatmap, **not raw sensor data** — see §4.2 for the required encoding. |
| `prompt` | optional | `str` | Defaults to `"Perform the task."` if omitted (`InjectDefaultPrompt`). Keep it fixed to this string — the checkpoint was trained on exactly this prompt for `cap_to_tray` (`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §1.4/§6, HF dataset card §8). |
| `action` / `actions` | **must be absent** | — | If present, `DeltaActions` (part of the same transform chain) will try to broadcast-subtract state against it and raise a shape error — it expects a full `(horizon, 32)` ground-truth chunk, which doesn't exist at inference time. Confirmed intentional: `canonical_tactile_policy.py:172-177`, "Actions are the training target and are absent at inference time." |

### 4.1 `observation.state` layout (32-D)

Same layout as training (`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.1,
`right_eef_mm_columns6d_10_19_revo2_raw6_20_26_v1`):

| Indices | Meaning | Units |
|---|---|---|
| `[0:10]` | unused (reserved) | zero |
| `[10:13]` | right arm EEF xyz | mm |
| `[13:19]` | right arm orientation, rot6d (first two columns of the rotation matrix) | — |
| `[19]` | unused | zero |
| `[20:26]` | 6 Revo2 hand motor targets, in the same order as the delivered dataset's `action/right/hand_target` | raw motor units, 0-1000 |
| `[26:32]` | unused (reserved) | zero |

This is the **last commanded** state — whatever arm/hand target you most
recently sent to the robot, not something read back from encoders. Convert
your live orientation reading to rot6d with
`n0vtla.policies.rotation_utils` (the same utility
`scripts/wetlab_smoke_adapter.py:decode_commands` uses to go the other
direction) — don't hand-roll the axis-angle-to-rot6d conversion; get the
column-vs-row convention right by using the shipped utility, not by matching
round-trip output to your own inverse (see the postmortem in
`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §2's rot6d row about why a self-consistent
round-trip test doesn't catch a systematic convention bug).

### 4.2 Tactile encoding — this is not raw sensor data

The model was trained on a **grayscale pressure-heatmap image**, not raw taxel
readings. Reproducing this at inference means, per taxel-pad reading:

1. Normalize: `(raw - baseline[pad]) / scale[pad]`, clipped to `[-1, 8]`
   (`itw_pressure.normalize_pressure`, reused as-is by
   `n0vtla/policies/canonical_tactile_policy.py`'s pipeline). Use the
   **per-pad `normal_baseline` / `normal_scale` from this project's own fitted
   `wetlab_tactile_norm_v2.json`** (train-split-only fit, restricted to each
   episode's own trial window — `scripts/fit_wetlab_tactile_norm.py`,
   `ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.8 item 3), **not** the human-corpus
   stats and not a fresh fit on your own live readings — the checkpoint's
   `norm_stats.json` and this pad-level file are two different things that
   both need to match training.
2. Map to grayscale: `gray = round((clip(normal,-1,8) + 1) * 255/9)`, replicate
   to 3 channels (`itw_pressure.pressure_rgb`).
2. Lay out all 15 pads into one 224×224 canvas at their fixed positions
   (`itw_tactile_smoke_adapter.TACTILE_SLOT_LAYOUT`, consumed via
   `itw_pressure._put_resized`) — a hand-shaped diagram, not a raster of the
   physical sensor grid.
4. **Do not treat "right_hand_data.npz" as the live glove file** — on this
   rig's raw data delivery the live signal was in the file named
   `left_hand_data.npz` (`ROBOT_POSTTRAIN_OPEN_ISSUES.md` §3.1). This was a
   **delivery labeling bug in the archived dataset**, not necessarily a
   property of your live sensor SDK — check your own hardware's channel
   ordering directly (e.g. via the guard pattern in
   `wetlab_smoke_adapter.resolve_episode_window`: whichever channel has
   `std >= 1e-4` is live) rather than assuming the same swap applies live.

**Baseline frame — read this carefully, it is not "the previous frame":**
Per training (`n0vtla/training/config.py:42`, `_LATENT_BASELINE_FRAMES =
100_000`, deliberately larger than any episode so the LeRobot delta-timestamp
loader clamps it to frame 0), the "baseline" tactile frame is the
**first frame of the episode**, paired with the current frame at every
subsequent step — not a rolling window, not the immediately-preceding frame.
Reproduce this exactly as `scripts/serve_zmq.py`'s own docstring specifies for
its (different, Flexiv-family) config, because the mechanism is identical:

> `reset` clears the stored baseline. Call it at the START of every episode.
> `predict` on the first call after a reset captures the current tactile as the
> baseline (so frame 0 has `tac_t == tac_0` → zero contact, matching
> training); every subsequent predict pairs the live tactile (`tac_t`) with
> that stored baseline (`tac_0`).

Concretely: send `observation.image.right_wrist_right_tactile` as a
`(2, H, W, 3)` stack `[baseline_frame, current_frame]` where `baseline_frame`
is the **same array, held fixed**, captured on your first `infer()` call after
`reset()`. If you'd rather send a single current-frame-only image (3D array,
no stack), `CanonicalTactileInputs` accepts that too, but then the baseline
slot is masked absent (`image_mask` false) and the model's contact signal
degrades to "no baseline available" — not what training saw. Get the stacked
version working before trusting any tactile-conditioned behavior.

`scripts/serve_zmq.py` is a complete reference implementation of this exact
reset/baseline pattern (for a sibling config family, not this one — its image
keys and state layout don't apply here, but its control flow does). Read it
before writing your own hardware-facing serve wrapper; don't reinvent the
reset semantics from scratch.

## 5. Response dict (what you get back)

`Policy.infer()` (`n0vtla/policies/policy.py:68-106`) returns:

```python
{
    "state": <your input state, echoed back>,
    "actions": np.ndarray,  # shape (horizon=50, 32), ABSOLUTE physical units
    "policy_timing": {"infer_ms": float},
}
```

`actions` is already fully denormalized and delta-inverted (`Unnormalize` +
`AbsoluteActions`, part of the policy's output transform chain) — decode it
with the **same layout as §4.1** (`[10:13]`=xyz mm absolute, `[13:19]`=rot6d
absolute, `[20:26]`=hand motor targets absolute 0-1000). Convert rot6d back to
whatever your arm controller wants (axis-angle, quaternion, ...) with
`n0vtla.policies.rotation_utils.rot6d_to_matrix`, the same function
`wetlab_smoke_adapter.decode_commands` uses — again, don't hand-roll this
conversion.

**Execute the full 50-step chunk before requesting the next prediction**
(DEPLOY.md) unless you deliberately implement receding-horizon replanning —
the model wasn't trained with a different replanning cadence in mind, and
`HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md`'s deployment gates explicitly recommend
receding-horizon replanning over blindly executing the entire chunk once you
have a working loop; that's a deliberate real-time control decision for
whoever integrates this, not something this contract prescribes either way.

## 6. Before commanding the real arm — safety gates

This document only specifies the *data contract*. It is not a green light to
move the robot. `HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md`'s "Post-Training and
Deployment Gates" section (written before this training run existed, still
applicable) lists the pre-motion checklist: physical-unit held-out action
error, smoothness/command-limit checks, causal sensor latency, source-action
replay through the exact deployed decoder — **before** any robot motion — and
then supervised low-speed deployment only, with workspace/joint/motor limits,
velocity/acceleration caps, stale-observation rejection, a watchdog, and an
accessible emergency stop. None of that is implemented by anything in this
repo or this document; it is rig-specific and must be built by whoever
operates the real xArm6 + Revo2.

## References

| Topic | File |
|---|---|
| Server entry point | `scripts/serve_policy.py` |
| Compatibility regression test | `scripts/gate_c_check.py` |
| Client library | `n0vtla_client/websocket_client_policy.py`, `n0vtla_client/base_policy.py` |
| The actual input contract (read this, not this doc, if they disagree) | `n0vtla/policies/canonical_tactile_policy.py` |
| Reference reset/baseline serve pattern (different config family, same mechanism) | `scripts/serve_zmq.py` |
| rot6d conversion utilities | `n0vtla.policies.rotation_utils` |
| Tactile normalization/encoding | `scripts/itw_pressure.py` (`normalize_pressure`, `pressure_rgb`), `scripts/itw_tactile_smoke_adapter.py` (`TACTILE_SLOT_LAYOUT`, `_put_resized`) |
| This checkpoint's provenance, training config, ship-gate results | `ROBOT_POSTTRAIN_QUICKSTART.md`, `ROBOT_POSTTRAIN_OPEN_ISSUES.md` |
| Pre-motion safety gates | `HUMAN_PRETRAIN_AND_ROBOT_POSTTRAIN.md`, "Post-Training and Deployment Gates" |
