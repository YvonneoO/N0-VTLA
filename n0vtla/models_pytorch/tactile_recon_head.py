"""TactileReconHead: r_psi, the auxiliary L1-reconstruction head from the paper's Stage 1
predictor-grounding objective (arXiv:2607.23782 Sec 4.2, Eq. 5):

    L_1 = L_NCE + lambda_rec * L_rec,   L_rec = || r_psi(z) - Dbar_{t->t+H} ||_1

This head and the Stage-1 training loop that uses it are NOT part of the open-sourced N0-VTLA
repo (the release ships only the action-conditioned post-training path; see
docs/STAGE1_PREDICTOR_PRETRAINING.md for the audit that established this). This module is our
own implementation of the paper's Sec 4.2 recipe, written to fill that gap so we can pretrain
the tactile predictor on data with no ground-truth robot actions (e.g. itw hand/glove tactile).

The paper does not publish r_psi's architecture or Dbar's resolution -- both are our own design
choices, called out here and in the design doc so they are easy to revisit:
  * Dbar lives in PIXEL space (not DINOv2 feature space): the per-view current->future tactile
    difference image, averaged over the active (real, non-placeholder) views, downsampled to a
    small (grid x grid) grid via adaptive average pooling. See N0VTLAPolicy._build_future_target.
  * r_psi is a 2-layer MLP over the mean-pooled z, chosen to mirror the paper's own description
    of the head as "a lightweight reconstruction head" -- no claim this matches NeoteAI's design.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TactileReconHead(nn.Module):
    """Decodes latent tactile tokens z -> a coarse (grid x grid) contact-change field.

    Args:
        hidden_dim: width of z's tokens (llm_dim, matching TactileActionPredictor's output).
        grid: side length of the coarse output field. Dbar is downsampled to the same size
            (see N0VTLAPolicy._build_future_target) so the L1 term compares equal shapes.
        mlp_ratio: hidden width multiplier for the 2-layer MLP.
    """

    def __init__(self, hidden_dim: int, grid: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.grid = grid
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, grid * grid),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n_latent, hidden_dim) -> (B, grid, grid)."""
        pooled = z.mean(dim=1)  # (B, hidden_dim) -- same pooling convention as the InfoNCE h(.)
        field = self.mlp(pooled)  # (B, grid*grid)
        return field.view(z.shape[0], self.grid, self.grid)
