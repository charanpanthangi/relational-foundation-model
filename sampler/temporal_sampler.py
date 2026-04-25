from __future__ import annotations

import pandas as pd


def sample_before_time(df: pd.DataFrame, timestamp_column: str, as_of: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df[timestamp_column], errors="coerce")
    return df.loc[ts <= as_of].copy()
