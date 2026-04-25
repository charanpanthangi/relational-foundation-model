from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv


class HGTBaseline(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, metadata: tuple[list[str], list[tuple[str, str, str]]], num_layers: int = 2, heads: int = 2) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.input_proj = nn.ModuleDict({nt: nn.LazyLinear(hidden_dim) for nt in node_types})
        self.layers = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=heads) for _ in range(num_layers)])
        self.out_proj = nn.ModuleDict({nt: nn.Linear(hidden_dim, out_dim) for nt in node_types})

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        h = {k: self.input_proj[k](v.float()) for k, v in x_dict.items()}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return {k: self.out_proj[k](v) for k, v in h.items()}
