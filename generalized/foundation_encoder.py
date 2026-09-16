from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NullFoundationEncoder(nn.Module):
    """Zero-feature fallback used when foundation features are disabled."""

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return images.new_zeros((images.shape[0], self.output_dim))

    @torch.no_grad()
    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        device = next(self.parameters(), torch.empty(0)).device
        return torch.zeros((len(class_names), self.output_dim), device=device)


class OpenAIClipEncoder(nn.Module):
    """Frozen CLIP image/text encoder for multimodal prompt refinement.

    The module deliberately keeps CLIP frozen. Only the lightweight prompt
    refiner is optimized, which avoids turning the experiment into a costly
    full foundation-model fine-tuning task.
    """

    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__()
        try:
            import clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required when foundation.enabled=true. "
                "Install requirements_generalized.txt or disable this module."
            ) from exc

        model, _ = clip.load(model_name, device=device, jit=False)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self._clip = clip
        self.output_dim = int(model.visual.output_dim)
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images.float(), size=(224, 224), mode="bicubic", align_corners=False)
        images = (images - self.clip_mean) / self.clip_std
        return images.to(dtype=next(self.model.parameters()).dtype)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(self._prepare_images(images))
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        prompts = ["a photo of {}".format(name.replace("_", " ")) for name in class_names]
        tokens = self._clip.tokenize(prompts).to(next(self.model.parameters()).device)
        features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)


def build_foundation_encoder(config: dict, device: torch.device) -> nn.Module:
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return NullFoundationEncoder(output_dim=int(config.get("fallback_dim", 512))).to(device)
    model_name = str(config.get("model_name", "ViT-B/32"))
    return OpenAIClipEncoder(model_name=model_name, device=device).to(device)
