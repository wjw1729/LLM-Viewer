"""Engram conditional-memory cost (DeepSeek-V4.1).

Per token, per module (inference/model.py Engram.forward):
  1. Gather (max_ngram_size-1) * n_heads embedding rows
  2. Linear wkv: concat(rows) -> one key per Hyper-Connection stream + one shared value
  3. RMS-normalized dot-product gate, then residual add into the hc streams

The embedding tables are host-resident and prefetched. Only the gathered rows
move; the full table is reported separately and is not charged as HBM weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ops.moe.base import MoeLayerCost


@dataclass
class EngramDims:
    batchsize: int
    tokens: int
    hidden_size: int
    hc_mult: int
    max_ngram_size: int
    n_heads: int
    head_dim: int
    proj_byte: float
    a_byte: float
    row_byte: float = 1.0  # FP8 table entry


def _n_cols(dims: EngramDims) -> int:
    return max(1, int(dims.max_ngram_size) - 1) * int(dims.n_heads)


def _weight_bytes(n_params: float, byte: float) -> float:
    """Data bytes plus one scale byte per 32 elements for FP8/FP4 storage."""
    data = n_params * byte
    if byte <= 1.0:
        data += n_params / 32.0
    return data


def engram_costs(dims: EngramDims) -> List[MoeLayerCost]:
    """Costs for one Engram module over `dims.tokens` tokens (one sequence length)."""
    tokens = dims.batchsize * max(1, int(dims.tokens))
    cols = _n_cols(dims)
    in_dim = cols * dims.head_dim
    out_dim = dims.hidden_size * (dims.hc_mult + 1)
    a = dims.a_byte
    hc = dims.hc_mult
    d = dims.hidden_size
    # One token gathers `in_dim` FP8 elements (plus per-32 scales).
    lookup_bytes = tokens * _weight_bytes(float(in_dim), dims.row_byte)

    proj_params = float(in_dim * out_dim)
    proj_ops = 2.0 * proj_params * tokens
    # gate: per-stream RMS of h and key, dot, signed-sqrt sigmoid, residual add
    gate_ops = tokens * hc * d * 8.0 + tokens * hc * 6.0

    return [
        MoeLayerCost(
            name="engram_lookup",
            OPs=0,
            load_weight=0,
            load_act=lookup_bytes,
            store_act=tokens * in_dim * a,
        ),
        MoeLayerCost(
            name="engram_proj",
            OPs=proj_ops,
            load_weight=_weight_bytes(proj_params, dims.proj_byte),
            load_act=tokens * in_dim * a,
            store_act=tokens * out_dim * a,
        ),
        MoeLayerCost(
            name="engram_gate",
            OPs=gate_ops,
            load_weight=hc * d * 4.0,  # fp32 q_weight * k_weight
            load_act=tokens * hc * d * a * 2,
            store_act=tokens * hc * d * a,
        ),
    ]


def engram_host_bytes(num_embeddings: List[int], head_dim: int, row_byte: float = 1.0) -> float:
    """Full FP8 table footprint (host memory, not HBM)."""
    rows = sum(int(n) for n in num_embeddings)
    return _weight_bytes(rows * int(head_dim), row_byte)
