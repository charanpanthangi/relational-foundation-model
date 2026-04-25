from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from data_connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class ParquetConnector(BaseConnector):
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        if self.connection is None:
            self.connection = duckdb.connect()

    def list_tables(self) -> list[str]:
        return [p.stem for p in self.directory.glob("*.parquet")]

    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        path = self.directory / f"{table_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        query = f"SELECT * FROM read_parquet('{path.as_posix()}')"
        if limit:
            query += f" LIMIT {int(limit)}"
        return self.read_query(query)

    def read_query(self, query: str) -> pd.DataFrame:
        self.connect()
        assert self.connection is not None
        return self.connection.execute(query).fetchdf()

    def get_schema(self) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for table in self.list_tables():
            df = self.read_table(table, limit=100)
            out[table] = {c: str(t) for c, t in df.dtypes.items()}
        return out

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None
