from __future__ import annotations

import torch


def build_batches(indices: list[int], batch_size: int) -> list[torch.Tensor]:
    return [torch.tensor(indices[i:i + batch_size], dtype=torch.long) for i in range(0, len(indices), batch_size)]
