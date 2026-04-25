from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def encode_features(df: pd.DataFrame) -> torch.Tensor:
    if df.empty:
        return torch.empty((0, 1), dtype=torch.float32)

    encoded = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            arr = s.fillna(0).to_numpy(dtype=np.float32)
        elif pd.api.types.is_datetime64_any_dtype(s):
            arr = (s.fillna(pd.Timestamp("1970-01-01")).view("int64") / 1e9).to_numpy(dtype=np.float32)
        else:
            arr = s.fillna("<NA>").astype("category").cat.codes.to_numpy(dtype=np.float32)
        encoded.append(arr)
    return torch.from_numpy(np.stack(encoded, axis=1))
