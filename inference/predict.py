from __future__ import annotations

import torch

from models.relational_transformer import RelationalTransformer


def predict_single(model: RelationalTransformer, x_dict: dict[str, torch.Tensor], edge_index_dict: dict, anchor_type: str, entity_idx: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict, anchor_type=anchor_type)
        if out.ndim == 2 and out.shape[1] > 1:
            prob = torch.softmax(out[entity_idx], dim=-1)[1].item()
            return {"score": prob}
        return {"score": float(out[entity_idx].item())}
