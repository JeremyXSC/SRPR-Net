from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class ContextResolver:
    """Resolve an image path to a semantic context such as dish, scene or domain."""

    def __init__(self, mapping_path: Optional[str], default_context: str = "default") -> None:
        self.default_context = str(default_context)
        self.mapping: Dict[str, str] = {}
        if mapping_path:
            raw = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
            self.mapping = {str(key): str(value) for key, value in raw.items()}

    def resolve(self, paths: Iterable[str]) -> List[str]:
        contexts = []
        for path in paths:
            key = str(path)
            name = Path(key).name
            stem = Path(key).stem
            contexts.append(
                self.mapping.get(key, self.mapping.get(name, self.mapping.get(stem, self.default_context)))
            )
        return contexts
