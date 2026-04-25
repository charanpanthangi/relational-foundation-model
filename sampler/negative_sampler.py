from __future__ import annotations

import numpy as np


def sample_negatives(positive_indices: list[int], total_size: int, num_samples: int) -> list[int]:
    forbidden = set(positive_indices)
    candidates = [i for i in range(total_size) if i not in forbidden]
    if not candidates:
        return []
    rng = np.random.default_rng(42)
    k = min(num_samples, len(candidates))
    return rng.choice(candidates, size=k, replace=False).tolist()
