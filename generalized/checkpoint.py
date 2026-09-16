from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_generalized_checkpoint(path: str, state: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)


def load_generalized_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)
