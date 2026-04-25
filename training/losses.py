from __future__ import annotations

import torch
from torch import nn


def get_loss(task_type: str) -> nn.Module:
    if task_type == "classification":
        return nn.CrossEntropyLoss()
    if task_type == "regression":
        return nn.MSELoss()
    return nn.BCEWithLogitsLoss()


def compute_loss(logits: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    loss_fn = get_loss(task_type)
    if task_type == "classification":
        return loss_fn(logits, target.long())
    return loss_fn(logits.squeeze(), target.float())
