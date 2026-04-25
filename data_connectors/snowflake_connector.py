from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from data_connectors.base import BaseConnector

logger = logging.getLogger(__name__)


@dataclass
class SnowflakeConfig:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str


class SnowflakeConnector(BaseConnector):
    """Connector scaffold following BaseConnector contract."""

    def __init__(self, config: SnowflakeConfig) -> None:
        self.config = config
        self._connected = False

    def connect(self) -> None:
        logger.info("Snowflake connector initialized for account=%s", self.config.account)
        self._connected = True

    def list_tables(self) -> list[str]:
        self.connect()
        return []

    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.schema}.{table_name}"
        if limit:
            query += f" LIMIT {int(limit)}"
        return self.read_query(query)

    def read_query(self, query: str) -> pd.DataFrame:
        self.connect()
        logger.warning("Snowflake runtime client not configured; returning empty frame for query: %s", query)
        return pd.DataFrame()

    def get_schema(self) -> dict[str, dict[str, str]]:
        return {}
