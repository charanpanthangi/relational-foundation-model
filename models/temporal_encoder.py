from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, hidden_dim)

    def forward(self, timestamp: torch.Tensor) -> torch.Tensor:
        # timestamp is unix seconds
        sin = torch.sin(timestamp / 86_400.0)
        cos = torch.cos(timestamp / 86_400.0)
        return self.proj(torch.stack([sin, cos], dim=-1))
