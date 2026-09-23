"""Shared types for Manifold-Constrained Hyper-Connections (mHC) cost models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


@dataclass
class HcDims:
    batchsize: int
    seqlen: int
    hidden_size: int
    hc_mult: int
    sinkhorn_iters: int
    w_byte: float
    a_byte: float


@dataclass
class HcLayerCost:
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


def mix_dim(hc_mult: int) -> int:
    """Output width of hc_fn: pre(n) + post(n) + comb(n*n)."""
    n = max(1, int(hc_mult))
    return (2 + n) * n


def hc_pre_cost(dims: HcDims, name: str = "hc_pre") -> HcLayerCost:
    """Reduce [B,S,n,D] residual streams to a single [B,S,D] branch input.

    Matches DeepSeek-V4: flatten -> RMSNorm scale -> Linear(nD -> mix) ->
    Sinkhorn on comb -> weighted sum over streams.
    """
    B, S, D = dims.batchsize, dims.seqlen, dims.hidden_size
    n = max(1, int(dims.hc_mult))
    mix = mix_dim(n)
    tokens = B * S
    nd = n * D
    a, w = dims.a_byte, dims.w_byte
    iters = max(1, int(dims.sinkhorn_iters))

    # RMSNorm over flattened nD (~7 ops/elem) + linear + reduce + sinkhorn
    norm_ops = tokens * nd * 7
    linear_ops = tokens * nd * mix * 2
    reduce_ops = tokens * n * D
    # softmax + row/col renorm per Sinkhorn iter (~10 ops per n*n entry)
    sinkhorn_ops = tokens * n * n * 10 * iters

    return HcLayerCost(
        name=name,
        OPs=norm_ops + linear_ops + reduce_ops + sinkhorn_ops,
        load_weight=mix * nd * w,
        load_act=tokens * nd * a,
        store_act=tokens * D * a + tokens * (n + n * n) * a,  # y + post + comb
    )


def hc_post_cost(dims: HcDims, name: str = "hc_post") -> HcLayerCost:
    """Write branch output back into n residual streams: post*x + comb@residual."""
    B, S, D = dims.batchsize, dims.seqlen, dims.hidden_size
    n = max(1, int(dims.hc_mult))
    tokens = B * S
    a = dims.a_byte

    post_ops = tokens * n * D
    comb_ops = tokens * n * n * D * 2

    return HcLayerCost(
        name=name,
        OPs=post_ops + comb_ops,
        load_weight=0,
        load_act=tokens * D * a + tokens * n * D * a,  # branch + residual
        store_act=tokens * n * D * a,
    )


def hc_head_cost(dims: HcDims, name: str = "hc_head") -> HcLayerCost:
    """Final HyperHead: contract [B,S,n,D] -> [B,S,D] before lm_head."""
    B, S, D = dims.batchsize, dims.seqlen, dims.hidden_size
    n = max(1, int(dims.hc_mult))
    tokens = B * S
    nd = n * D
    a, w = dims.a_byte, dims.w_byte

    norm_ops = tokens * nd * 7
    linear_ops = tokens * nd * n * 2  # hc_head_fn: [n, nD]
    reduce_ops = tokens * n * D

    return HcLayerCost(
        name=name,
        OPs=norm_ops + linear_ops + reduce_ops,
        load_weight=n * nd * w,
        load_act=tokens * nd * a,
        store_act=tokens * D * a,
    )


def embedding_cost(
    batchsize: int,
    seqlen: int,
    hidden_size: int,
    vocab_size: int,
    w_byte: float,
    a_byte: float,
    name: str = "embedding",
) -> HcLayerCost:
    """Token embedding gather (memory-bound; OPs≈0)."""
    tokens = batchsize * max(1, int(seqlen))
    return HcLayerCost(
        name=name,
        OPs=0,
        load_weight=vocab_size * hidden_size * w_byte,
        load_act=tokens * a_byte,  # token ids
        store_act=tokens * hidden_size * a_byte,
    )


def block_hc_costs(dims: HcDims) -> List[HcLayerCost]:
    """Per transformer block: attn + ffn Hyper-Connections (replaces residual adds)."""
    return [
        hc_pre_cost(dims, "attn_hc_pre"),
        hc_post_cost(dims, "attn_hc_post"),
        hc_pre_cost(dims, "ffn_hc_pre"),
        hc_post_cost(dims, "ffn_hc_post"),
    ]
