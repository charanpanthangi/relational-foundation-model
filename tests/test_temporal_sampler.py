import pandas as pd

from sampler.temporal_sampler import sample_before_time


def test_temporal_sample() -> None:
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-02-01"], "v": [1, 2]})
    out = sample_before_time(df, "ts", pd.Timestamp("2024-01-15"))
    assert out["v"].tolist() == [1]
