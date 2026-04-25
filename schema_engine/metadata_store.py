from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetadataStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())
