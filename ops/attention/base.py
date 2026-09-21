"""Shared types and base class for attention cost models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class AttnDims:
    """Common tensor / dtype dims passed into attention cost models."""

    batchsize: int
    seqlen: int
    num_attention_heads: int
    num_key_value_heads: int
    head_size: int
    a_byte: float
    kv_byte: float
    onchip_buffer: float


@dataclass
class AttnLayerCost:
    """One named sub-op cost (qk_matmul / softmax / fused_attention / ...)."""

    name: str
    OPs: float = 0
    load_weight: float = 0
    load_act: float = 0
    store_act: float = 0
    load_kv_cache: float = 0
    store_kv_cache: float = 0

    def as_kwargs(self, stage: str) -> dict:
        """Flatten into ModelAnalyzer._analyze_to_results kwargs."""
        d = asdict(self)
        d["stage"] = stage
        return d


class BaseAttentionOp(ABC):
    """Extensible attention cost model (full / SWA / CSA / ...)."""

    kind: str = "base"

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size

    @abstractmethod
    def decode_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        raise NotImplementedError

    @abstractmethod
    def prefill_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        raise NotImplementedError

    def kv_cache_tokens(self, seqlen: int) -> int:
        """Tokens retained in KV cache for memory-footprint estimates."""
        return seqlen

    def effective_kv_len(self, seqlen: int) -> int:
        """Keys/values attended per query (may be < seqlen for windowed attn)."""
        return seqlen
