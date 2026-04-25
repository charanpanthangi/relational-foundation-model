from __future__ import annotations

import pandas as pd


def build_temporal_index(df: pd.DataFrame, timestamp_column: str) -> pd.Series:
    if timestamp_column not in df:
        raise KeyError(f"Missing timestamp column: {timestamp_column}")
    s = pd.to_datetime(df[timestamp_column], errors="coerce")
    return s.sort_values().reset_index(drop=True)
