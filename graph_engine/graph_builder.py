from __future__ import annotations

from typing import Any

import pandas as pd
from torch_geometric.data import HeteroData

from data_connectors.base import BaseConnector
from graph_engine.edge_builder import build_edges
from graph_engine.feature_encoder import encode_features
from graph_engine.hetero_graph import GraphPayload
from schema_engine.relationship_detector import detect_relationships


def build_hetero_graph(connector: BaseConnector, max_rows_per_table: int | None = None) -> GraphPayload:
    schema = connector.get_schema()
    frames: dict[str, pd.DataFrame] = {
        table: connector.read_table(table, limit=max_rows_per_table)
        for table in schema
    }

    data = HeteroData()
    for table, frame in frames.items():
        data[table].x = encode_features(frame)

    for rel in detect_relationships(schema):
        src_df, dst_df = frames[rel.source_table], frames[rel.target_table]
        edge_idx = build_edges(src_df, dst_df, rel)
        edge_type = (rel.source_table, f"to_{rel.target_table}", rel.target_table)
        data[edge_type].edge_index = edge_idx

    data.metadata_cache = {"tables": list(schema.keys())}  # type: ignore[attr-defined]
    return GraphPayload(data=data, table_frames=frames)
