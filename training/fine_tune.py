from __future__ import annotations

import torch

from training.train import train


def fine_tune(checkpoint_path: str) -> dict[str, float]:
    if checkpoint_path:
        _ = torch.load(checkpoint_path, map_location="cpu")
    return train()
