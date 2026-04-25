from __future__ import annotations

from data_connectors.base import BaseConnector


def introspect_schema(connector: BaseConnector) -> dict[str, dict[str, str]]:
    return connector.get_schema()
