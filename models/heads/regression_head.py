from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)
