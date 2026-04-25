from __future__ import annotations

import torch
from torch import nn


class TaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_dim)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_ids)
