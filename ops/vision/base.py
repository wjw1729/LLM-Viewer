"""Vision encoder cost models (ViT + aligner), separate from LLM ops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class VisionDims:
    batchsize: int
    num_images: int
    n_vit_tokens: int  # patches before spatial downsample
    n_out_tokens: int  # tokens after aligner (into LLM)
    vision_dim: int
    vision_n_heads: int
    vision_inter_dim: int
    vision_n_layers: int
    vision_patch_size: int
    downsample_ratio: int
    llm_hidden_size: int
    w_byte: float
    a_byte: float


@dataclass
class VisionLayerCost:
    name: str
    OPs: float = 0
    load_weight: float = 0
    load_act: float = 0
    store_act: float = 0
    load_kv_cache: float = 0
    store_kv_cache: float = 0

    def as_kwargs(self, stage: str = "prefill") -> dict:
        d = asdict(self)
        d["stage"] = stage
        return d


class BaseVisionOp(ABC):
    kind: str = "base"

    @abstractmethod
    def prefill_costs(self, dims: VisionDims) -> List[VisionLayerCost]:
        """Vision path runs once at prefill (not per decode step)."""
        raise NotImplementedError
