from __future__ import annotations

import torch
from torch import nn

from models.hgt_baseline import HGTBaseline
from models.heads.classification_head import ClassificationHead
from models.heads.link_prediction_head import LinkPredictionHead
from models.heads.regression_head import RegressionHead


class RelationalTransformer(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, metadata: tuple[list[str], list[tuple[str, str, str]]], task_type: str = "classification") -> None:
        super().__init__()
        self.encoder = HGTBaseline(hidden_dim=hidden_dim, out_dim=out_dim, metadata=metadata)
        self.task_type = task_type
        if task_type == "classification":
            self.head: nn.Module = ClassificationHead(out_dim)
        elif task_type == "regression":
            self.head = RegressionHead(out_dim)
        else:
            self.head = LinkPredictionHead(out_dim)

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor], anchor_type: str, pair_dst: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(x_dict, edge_index_dict)
        anchor = encoded[anchor_type]
        if self.task_type == "ranking":
            if pair_dst is None:
                raise ValueError("pair_dst required for ranking")
            return self.head(anchor, pair_dst)
        return self.head(anchor)
