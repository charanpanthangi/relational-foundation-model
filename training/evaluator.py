from __future__ import annotations

import numpy as np
import torch

from training.metrics import classification_metrics, regression_metrics


def evaluate_predictions(pred: torch.Tensor, target: torch.Tensor, task_type: str) -> dict[str, float]:
    if task_type == "classification":
        probs = torch.softmax(pred, dim=-1)[:, 1].detach().cpu().numpy()
        y = target.detach().cpu().numpy()
        return classification_metrics(y, probs)
    yhat = pred.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    return regression_metrics(y, yhat)
