"""Vision encoder cost-model package."""

from __future__ import annotations

from typing import Optional

from ops.vision.base import VisionDims, VisionLayerCost, BaseVisionOp
from ops.vision.deepseek_vit import DeepSeekViTOp

__all__ = [
    "VisionDims",
    "VisionLayerCost",
    "BaseVisionOp",
    "DeepSeekViTOp",
    "create_vision_op",
    "vision_token_geometry",
]


def vision_token_geometry(
    n_out_tokens: int,
    downsample_ratio: int = 3,
) -> tuple:
    """Map aligner output tokens -> ViT patch count.

    Aligner does R x R pixel-unshuffle, so n_vit ≈ n_out * R^2.
    """
    r = max(1, int(downsample_ratio))
    n_out = max(1, int(n_out_tokens))
    n_vit = n_out * r * r
    return n_vit, n_out


def create_vision_op(kind: str = "deepseek_vit") -> BaseVisionOp:
    key = (kind or "deepseek_vit").lower()
    if key in ("deepseek_vit", "deepseek", "vit"):
        return DeepSeekViTOp()
    raise ValueError(f"Unknown vision op kind {kind!r}")
