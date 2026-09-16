from __future__ import annotations

import random
from typing import Any, Dict, Iterable

import numpy as np
import torch


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Whole-checkpoint loading commonly uses ``map_location=device``.  That
    # also moves RNG byte tensors to CUDA, while both CPU and CUDA generator
    # restoration APIs require CPU uint8 state tensors.
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu")
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            torch.as_tensor(cuda_state, dtype=torch.uint8, device="cpu")
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def create_optimizer(
    parameters: Iterable,
    config: Dict[str, Any],
    lr: float,
    weight_decay: float,
):
    name = str(config.get("optimizer", "adamw")).lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(lr),
            momentum=float(config.get("momentum", 0.937)),
            weight_decay=float(weight_decay),
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
    raise ValueError("Unsupported optimizer: {}".format(name))


def create_scheduler(optimizer, config: Dict[str, Any], epochs: int):
    name = str(config.get("scheduler", "cosine")).lower()
    if name == "linear":
        final_factor = float(config.get("final_lr_factor", 0.1))

        def schedule(epoch: int) -> float:
            denominator = max(int(epochs) - 1, 1)
            progress = min(max(float(epoch) / denominator, 0.0), 1.0)
            return 1.0 - (1.0 - final_factor) * progress

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
        )
    raise ValueError("Unsupported scheduler: {}".format(name))
