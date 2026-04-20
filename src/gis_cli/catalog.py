from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IntentDefinition:
    name: str
    label: str
    keywords: list[str]
    risk: str
    steps: list[str]


class IntentCatalog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("config") / "intents.json"
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.version = payload["version"]
        self.intents = [IntentDefinition(**x) for x in payload["intents"]]

    def all(self) -> list[IntentDefinition]:
        return self.intents
