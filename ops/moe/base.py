"""Shared types and base class for MoE cost models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class MoeDims:
    batchsize: int
    seqlen: int
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    w_byte: float
    a_byte: float


@dataclass
class MoeLayerCost:
    name: str
    OPs: float = 0
    load_weight: float = 0
    load_act: float = 0
    store_act: float = 0
    load_kv_cache: float = 0
    store_kv_cache: float = 0

    def as_kwargs(self, stage: str) -> dict:
        d = asdict(self)
        d["stage"] = stage
        return d


def expert_weight_bytes(dims: MoeDims) -> float:
    """One SwiGLU expert: gate + up + down."""
    h, i = dims.hidden_size, dims.moe_intermediate_size
    return (h * i * 2 + i * h) * dims.w_byte


def n_active_experts(dims: MoeDims) -> int:
    return dims.num_experts_per_tok + dims.n_shared_experts


def experts_cost(dims: MoeDims, tokens: int, name: str = "experts") -> MoeLayerCost:
    """Activated experts compute/bandwidth for `tokens` tokens."""
    h, i = dims.hidden_size, dims.moe_intermediate_size
    n_active = n_active_experts(dims)
    # 3 matmuls (2 flop each) + silu*mul on intermediate
    ops_per_token = n_active * (6 * h * i + 2 * i)
    return MoeLayerCost(
        name=name,
        OPs=ops_per_token * tokens,
        load_weight=n_active * expert_weight_bytes(dims),
        load_act=h * tokens * dims.a_byte,
        store_act=h * tokens * dims.a_byte,
    )


class BaseMoeOp(ABC):
    """Extensible MoE routing + experts cost model."""

    kind: str = "base"

    def __init__(self, name_prefix: str = ""):
        # e.g. "" -> router/experts; "hash_" -> hash_router/hash_experts
        self.name_prefix = name_prefix

    def _name(self, base: str) -> str:
        return f"{self.name_prefix}{base}"

    @abstractmethod
    def decode_costs(self, dims: MoeDims) -> List[MoeLayerCost]:
        raise NotImplementedError

    @abstractmethod
    def prefill_costs(self, dims: MoeDims) -> List[MoeLayerCost]:
        raise NotImplementedError

    def full_experts_weight_bytes(self, dims: MoeDims) -> float:
        """HBM-resident expert weights (all routed + shared)."""
        return (dims.n_routed_experts + dims.n_shared_experts) * expert_weight_bytes(dims)

    def active_experts_weight_bytes(self, dims: MoeDims) -> float:
        return n_active_experts(dims) * expert_weight_bytes(dims)
