import torch
from torch_geometric.data import HeteroData

from models.relational_transformer import RelationalTransformer


def test_model_forward() -> None:
    data = HeteroData()
    data["customer"].x = torch.randn(3, 4)
    data["order"].x = torch.randn(4, 4)
    data["order", "to_customer", "customer"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    model = RelationalTransformer(hidden_dim=8, out_dim=8, metadata=data.metadata(), task_type="classification")
    out = model(data.x_dict, data.edge_index_dict, anchor_type="customer")
    assert out.shape[0] == 3
