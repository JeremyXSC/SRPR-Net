from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPriorTable:
    """Context-conditioned class prior table.

    JSON format:
    {
      "class_names": ["person"],
      "default": [1.0],
      "contexts": {"street": [1.0]}
    }
    """

    def __init__(self, class_names: List[str], default: torch.Tensor, contexts: Dict[str, torch.Tensor]) -> None:
        self.class_names = list(class_names)
        self.default = self._normalize(default.float())
        self.contexts = {key: self._normalize(value.float()) for key, value in contexts.items()}

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp_min(1e-8)
        return values / values.sum().clamp_min(1e-8)

    @classmethod
    def from_json(cls, path: Optional[str], class_names: List[str]) -> "SemanticPriorTable":
        if not path:
            uniform = torch.ones(len(class_names), dtype=torch.float32)
            return cls(class_names, uniform, {})
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stored_names = data.get("class_names", class_names)
        if list(stored_names) != list(class_names):
            raise ValueError(
                "Semantic-prior class_names do not match dataset names: {} != {}".format(stored_names, class_names)
            )
        default = torch.tensor(data.get("default", [1.0] * len(class_names)), dtype=torch.float32)
        contexts = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in data.get("contexts", {}).items()
        }
        return cls(class_names, default, contexts)

    def batch(self, context_ids: Iterable[str], device: torch.device) -> torch.Tensor:
        values = [self.contexts.get(str(context_id), self.default) for context_id in context_ids]
        return torch.stack(values, dim=0).to(device)


class BayesianSemanticCalibrator(nn.Module):
    """Bayesian soft calibration of YOLO class probabilities.

    For proposal i and class c:
        q_ic proportional to p_ic * P(c | context)^alpha

    The class mass of each proposal is preserved, so calibration changes the
    relative class ranking rather than blindly inflating all confidences.
    """

    def __init__(
        self,
        prior_table: SemanticPriorTable,
        initial_alpha: float = 0.35,
        learnable_alpha: bool = True,
        max_alpha: float = 2.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.prior_table = prior_table
        self.max_alpha = float(max_alpha)
        self.enabled = bool(enabled)
        initial_alpha = min(max(float(initial_alpha), 1e-4), self.max_alpha - 1e-4)
        raw = torch.logit(torch.tensor(initial_alpha / self.max_alpha))
        self.raw_alpha = nn.Parameter(raw, requires_grad=bool(learnable_alpha))

    @property
    def alpha(self) -> torch.Tensor:
        if not self.enabled:
            return self.raw_alpha.new_zeros(())
        return self.max_alpha * torch.sigmoid(self.raw_alpha)

    def forward(self, prediction: torch.Tensor, context_ids: List[str], nc: int) -> torch.Tensor:
        if not self.enabled or nc <= 1:
            return prediction
        if prediction.ndim != 3:
            raise ValueError("YOLO prediction must have shape [B, N, C].")
        priors = self.prior_table.batch(context_ids, prediction.device)[:, None, :]
        class_prob = prediction[..., 5:5 + nc].clamp(1e-7, 1 - 1e-7)
        original_peak = class_prob.max(dim=-1, keepdim=True).values
        posterior = class_prob * priors.pow(self.alpha)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        posterior = posterior / posterior.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
        output = prediction.clone()
        output[..., 5:5 + nc] = posterior * original_peak
        return output

    def alignment_loss(self, prediction: torch.Tensor, context_ids: List[str], nc: int) -> torch.Tensor:
        if not self.enabled or nc <= 1:
            return prediction.sum() * 0.0
        class_prob = prediction[..., 5:5 + nc].clamp_min(1e-8)
        visual = class_prob.mean(dim=1)
        visual = visual / visual.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        prior = self.prior_table.batch(context_ids, prediction.device).clamp_min(1e-8)
        mixture = 0.5 * (visual + prior)
        js = 0.5 * (
            F.kl_div(mixture.log(), visual, reduction="batchmean")
            + F.kl_div(mixture.log(), prior, reduction="batchmean")
        )
        return js
