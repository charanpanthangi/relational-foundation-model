from __future__ import annotations

from collections import deque
from time import time

import torch

from inference.predict import predict_single
from models.relational_transformer import RelationalTransformer


class OnlinePredictor:
    def __init__(self, model: RelationalTransformer, max_events: int = 2000) -> None:
        self.model = model
        self.events: deque[dict[str, float]] = deque(maxlen=max_events)

    def score(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict, anchor_type: str, entity_idx: int) -> dict[str, float]:
        result = predict_single(self.model, x_dict, edge_index_dict, anchor_type, entity_idx)
        result["scored_at"] = time()
        self.events.append(result)
        return result
