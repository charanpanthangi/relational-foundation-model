from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseConnector(ABC):
    """Abstract interface for all relational connectors."""

    @abstractmethod
    def connect(self) -> None:
        """Create a live connection/session."""

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return accessible table names."""

    @abstractmethod
    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        """Load table into DataFrame."""

    @abstractmethod
    def read_query(self, query: str) -> pd.DataFrame:
        """Execute SQL and return DataFrame."""

    @abstractmethod
    def get_schema(self) -> dict[str, dict[str, str]]:
        """Return schema mapping of table->column->dtype."""

    def close(self) -> None:
        """Optional close hook."""

    def __enter__(self) -> "BaseConnector":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
