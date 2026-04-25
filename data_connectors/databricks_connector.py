from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from data_connectors.base import BaseConnector

logger = logging.getLogger(__name__)


@dataclass
class DatabricksConfig:
    server_hostname: str
    http_path: str
    access_token: str
    catalog: str = "main"
    schema: str = "default"


class DatabricksConnector(BaseConnector):
    def __init__(self, config: DatabricksConfig) -> None:
        self.config = config
        self._connected = False

    def connect(self) -> None:
        logger.info("Databricks SQL warehouse configured: %s", self.config.server_hostname)
        self._connected = True

    def list_tables(self) -> list[str]:
        self.connect()
        return []

    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {self.config.catalog}.{self.config.schema}.{table_name}"
        if limit:
            query += f" LIMIT {int(limit)}"
        return self.read_query(query)

    def read_query(self, query: str) -> pd.DataFrame:
        self.connect()
        logger.warning("Databricks driver not installed/configured; returning empty frame for query: %s", query)
        return pd.DataFrame()

    def get_schema(self) -> dict[str, dict[str, str]]:
        return {}
