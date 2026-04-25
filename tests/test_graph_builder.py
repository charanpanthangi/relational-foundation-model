from pathlib import Path

import pandas as pd

from data_connectors.parquet_connector import ParquetConnector
from graph_engine.graph_builder import build_hetero_graph


def test_graph_builder(tmp_path: Path) -> None:
    customers = pd.DataFrame({"id": [1, 2], "age": [20, 40]})
    orders = pd.DataFrame({"id": [10, 11], "customer_id": [1, 2], "amount": [5.0, 8.0]})
    customers.to_parquet(tmp_path / "customer.parquet")
    orders.to_parquet(tmp_path / "orders.parquet")

    graph = build_hetero_graph(ParquetConnector(str(tmp_path)))
    assert "customer" in graph.data.node_types
    assert "orders" in graph.data.node_types
    assert graph.data["orders", "to_customer", "customer"].edge_index.shape[0] == 2
