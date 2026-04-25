from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from torch_geometric.data import HeteroData


@dataclass
class GraphPayload:
    data: HeteroData
    table_frames: dict[str, pd.DataFrame]
