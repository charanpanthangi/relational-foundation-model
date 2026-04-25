from __future__ import annotations


def detect_timestamp_columns(schema: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for table, cols in schema.items():
        ts_cols = [
            c for c, t in cols.items() if any(k in c.lower() for k in ["time", "date", "timestamp", "at"])
            or any(k in t.lower() for k in ["date", "time"])
        ]
        matches[table] = ts_cols
    return matches
