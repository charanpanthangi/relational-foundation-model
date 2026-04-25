from __future__ import annotations

import torch
from torch import nn


class LinkPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.scorer(src, dst).squeeze(-1)
