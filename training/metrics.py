from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse}


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    topk = np.argsort(scores)[::-1][:k]
    positives = labels[topk].sum()
    total_pos = max(labels.sum(), 1)
    return float(positives / total_pos)
