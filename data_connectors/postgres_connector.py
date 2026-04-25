from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from data_connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class PostgresConnector(BaseConnector):
    def __init__(self, uri: str, schema: str = "public") -> None:
        self.uri = uri
        self.schema = schema
        self.engine: Engine | None = None

    def connect(self) -> None:
        if self.engine is None:
            logger.info("Connecting to Postgres %s", self.uri)
            self.engine = create_engine(self.uri, future=True)

    def list_tables(self) -> list[str]:
        self.connect()
        assert self.engine is not None
        return inspect(self.engine).get_table_names(schema=self.schema)

    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        sql = f"SELECT * FROM {self.schema}.{table_name}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.read_query(sql)

    def read_query(self, query: str) -> pd.DataFrame:
        self.connect()
        assert self.engine is not None
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def get_schema(self) -> dict[str, dict[str, str]]:
        self.connect()
        assert self.engine is not None
        insp = inspect(self.engine)
        out: dict[str, dict[str, str]] = {}
        for table in self.list_tables():
            cols = insp.get_columns(table, schema=self.schema)
            out[table] = {c["name"]: str(c["type"]) for c in cols}
        return out

    def close(self) -> None:
        if self.engine:
            self.engine.dispose()
            self.engine = None
