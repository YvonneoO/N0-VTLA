"""Print the learned tactile gate (z_gate) and z_proj statistics of checkpoints: how much the tactile branch can influence the action expert.
Usage: python scripts/inspect_z_gate.py CKPT_DIR [CKPT_DIR ...]"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

for d in sys.argv[1:]:
    with safe_open(str(Path(d) / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if "z_gate" in k:
                t = f.get_tensor(k).float()
                print(f"{d}: {k} shape {tuple(t.shape)} value(s) {t.flatten()[:6].tolist()}  |mean| {t.abs().mean():.5f}")
            elif k.endswith("z_proj.weight") or k.endswith("z_proj.0.weight"):
                t = f.get_tensor(k).float()
                print(f"{d}: {k} shape {tuple(t.shape)} |w| mean {t.abs().mean():.5f}")
