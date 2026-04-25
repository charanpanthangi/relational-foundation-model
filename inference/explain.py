from __future__ import annotations

import torch


def explain_linear_contribution(feature_vector: torch.Tensor, weight_vector: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
    contrib = feature_vector * weight_vector
    values, idx = torch.topk(contrib.abs(), k=min(top_k, contrib.numel()))
    return [(int(i.item()), float(contrib[i].item())) for i in idx]
