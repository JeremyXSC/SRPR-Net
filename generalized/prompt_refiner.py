from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MultiModalBoxRefiner(nn.Module):
    """Transformer-attention prompt refiner with image/text foundation features."""

    def __init__(
        self,
        foundation_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_center_shift: float = 0.25,
        max_log_scale: float = 0.35,
        enabled: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.max_center_shift = float(max_center_shift)
        self.max_log_scale = float(max_log_scale)
        self.enabled = bool(enabled)
        self.use_attention = bool(use_attention)

        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.image_projection = nn.Linear(foundation_dim, hidden_dim)
        self.text_projection = nn.Linear(foundation_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = (
            nn.TransformerEncoder(layer, num_layers=num_layers)
            if self.use_attention and num_layers > 0
            else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.quality_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _normalize_geometry(boxes: torch.Tensor, scores: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        height, width = image_hw
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        box_w = (x2 - x1).clamp_min(1.0)
        box_h = (y2 - y1).clamp_min(1.0)
        cx = (x1 + x2) * 0.5 / float(width)
        cy = (y1 + y2) * 0.5 / float(height)
        nw = box_w / float(width)
        nh = box_h / float(height)
        return torch.stack((cx, cy, nw, nh, scores.clamp(0.0, 1.0)), dim=-1)

    def _apply_delta(self, boxes: torch.Tensor, delta: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        height, width = image_hw
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        box_w = (x2 - x1).clamp_min(2.0)
        box_h = (y2 - y1).clamp_min(2.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        delta = torch.tanh(delta)
        cx = cx + delta[:, 0] * self.max_center_shift * box_w
        cy = cy + delta[:, 1] * self.max_center_shift * box_h
        box_w = box_w * torch.exp(delta[:, 2] * self.max_log_scale)
        box_h = box_h * torch.exp(delta[:, 3] * self.max_log_scale)

        raw_x1 = cx - 0.5 * box_w
        raw_y1 = cy - 0.5 * box_h
        raw_x2 = cx + 0.5 * box_w
        raw_y2 = cy + 0.5 * box_h
        x1 = raw_x1.clamp(0.0, float(max(width - 3, 0)))
        y1 = raw_y1.clamp(0.0, float(max(height - 3, 0)))
        x2 = torch.maximum(raw_x2.clamp(0.0, float(width - 1)), x1 + 2.0)
        y2 = torch.maximum(raw_y2.clamp(0.0, float(height - 1)), y1 + 2.0)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        image_feature: torch.Tensor,
        class_features: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            return boxes, scores.new_zeros((0,))
        if not self.enabled:
            return boxes, scores.detach().clamp(0.0, 1.0)
        geometry = self._normalize_geometry(boxes, scores, image_hw)
        tokens = self.geometry_encoder(geometry)
        tokens = tokens + self.image_projection(image_feature).view(1, -1)
        safe_labels = labels.long().clamp(0, max(class_features.shape[0] - 1, 0))
        tokens = tokens + self.text_projection(class_features[safe_labels])
        attended = self.attention(tokens.unsqueeze(0)).squeeze(0)
        delta = self.delta_head(attended)
        quality = torch.sigmoid(self.quality_head(attended).squeeze(-1))
        return self._apply_delta(boxes, delta, image_hw), quality
