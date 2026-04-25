from __future__ import annotations

from typing import Iterable

import torch

from inference.predict import predict_single
from models.relational_transformer import RelationalTransformer


def batch_predict(model: RelationalTransformer, x_dict: dict[str, torch.Tensor], edge_index_dict: dict, anchor_type: str, entity_ids: Iterable[int]) -> list[dict[str, float]]:
    return [predict_single(model, x_dict, edge_index_dict, anchor_type, i) for i in entity_ids]
