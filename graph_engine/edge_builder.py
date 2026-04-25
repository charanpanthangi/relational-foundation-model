from __future__ import annotations

import pandas as pd
import torch

from schema_engine.relationship_detector import Relationship


def build_edges(source_df: pd.DataFrame, target_df: pd.DataFrame, rel: Relationship) -> torch.Tensor:
    if rel.source_column not in source_df or rel.target_column not in target_df:
        return torch.empty((2, 0), dtype=torch.long)

    target_index = {k: i for i, k in enumerate(target_df[rel.target_column].tolist())}
    rows = []
    for src_idx, target_key in enumerate(source_df[rel.source_column].tolist()):
        if target_key in target_index:
            rows.append((src_idx, target_index[target_key]))
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long).T
