from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
