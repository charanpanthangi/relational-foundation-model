from __future__ import annotations

import pandas as pd


def sample_entity_context(df: pd.DataFrame, entity_column: str, entity_id: int | str, limit: int = 200) -> pd.DataFrame:
    subset = df[df[entity_column] == entity_id]
    return subset.head(limit).copy()
