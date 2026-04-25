from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Relationship:
    source_table: str
    source_column: str
    target_table: str
    target_column: str = "id"


def detect_relationships(schema: dict[str, dict[str, str]]) -> list[Relationship]:
    table_names = set(schema)
    relationships: list[Relationship] = []
    for table, cols in schema.items():
        for col in cols:
            if not col.endswith("_id"):
                continue
            candidate = col[:-3]
            target = candidate if candidate in table_names else f"{candidate}s"
            if target in table_names:
                relationships.append(Relationship(table, col, target, "id"))
    return relationships
