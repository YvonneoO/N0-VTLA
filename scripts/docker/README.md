# N0-VTLA training image

Trains Stage-2 → post-train on top of our already-trained Stage-1 checkpoint. You
don't need to run Stage-1 or build anything — just pull the image and run two scripts.

## Steps

```bash
# 1. Edit the 4 placeholders at the top of open_container.sh:
#    DATA_DIR, CHECKPOINTS_DIR, ASSETS_DIR, HF_TOKEN -- set them to real local
#    paths / your HF token.

# 2. Pull the image, then run open_container.sh to open a shell inside it
#    (open_container.sh does this pull itself -- shown here so you can run it
#    ahead of time / verify network access first):
docker pull qqyang/n0vtla_train:latest
bash open_container.sh

# 3. Inside the container, run:
bash scripts/docker/quickstart.sh
```

That's it. When it finishes, it prints where the final checkpoint is on your machine (under the `CHECKPOINTS_DIR` you set) — upload to hugging face so we can download it. Thanks for your help!

To try it on a tiny scale first: `bash scripts/docker/quickstart.sh --num-train-steps=5`

## Reference (only if something goes wrong, or you want manual control)

`quickstart.sh` runs `setup_stage2_prereqs.sh` (downloads: base checkpoint, our
Stage-1 checkpoint, a Stage-2 data slice, and the post-train dataset) then
`run_stage2_onward.sh` (Stage-2 train → merge → post-train). Both accept env var
overrides (`STAGE1_STEP`, `OPENNEODATA_PLATFORM`, `EXP_NAME_STAGE2`, etc. — see each
script's top) and pass extra CLI args through to the underlying training calls.

### Env vars


| Var                                   | Used by                        | Default                           |
| ------------------------------------- | ------------------------------ | --------------------------------- |
| `VTLA_STAGE1_CHECKPOINT`              | Stage-2                        | `checkpoints/n0-vtla_ts_pretrain` |
| `VTLA_DATASET_PATH` / `VTLA_ASSET_ID` | Stage-2, post-train            | set by the scripts                |
| `VTLA_PRETRAINED_CHECKPOINT`          | all                            | set by the scripts                |
| `HF_TOKEN`                            | Stage-2's OpenNeoData download | required                          |
| `CHECK_ONLY=1`                        | any `train_*.sh`               | preflight only, no training       |
| `NPROC_PER_NODE`                      | all                            | auto-detected from `nvidia-smi`, falls back to 8 |


### Troubleshooting

- `Could not load libtorchcodec` — FFmpeg shared libs; already baked into the image.
- **DINOv2 cache miss** — don't override `HF_HOME`.
- **"A full epoch had no valid future tactile targets"** — a real data QC-gate failure,
not a code bug (not expected on this flow since data comes from our own scripts).
- **Multi-GPU crash with `NCCL... illegal memory access`** — usually a P2P/NVLink
  compatibility issue between GPUs on that specific machine, not a code bug (hit this
  ourselves on 4 GPUs; the DDP code itself is correct and tested at up to 4 GPUs). Add
  `-e NCCL_P2P_DISABLE=1` to the `docker run` command (forces GPU-to-GPU traffic through
  a slower but more compatible path) — this alone resolved it for us.

Not included: Stage-1 training (we already ran it), offline eval (not needed — just send back the post-train checkpoint).