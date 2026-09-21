"""Shared helpers for compressed-KV attention (CSA / HCA)."""

from __future__ import annotations

import math
from typing import List

from ops.attention.base import AttnDims, AttnLayerCost


def n_compressed_tokens(seqlen: int, compress_ratio: int) -> int:
    """Non-overlap pool length: ceil(S / R), at least 1."""
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be > 0, got {compress_ratio}")
    return max(1, math.ceil(seqlen / compress_ratio))


def attention_over_kv(
    dims: AttnDims,
    q_tokens: int,
    kv_tokens: int,
    use_flashattention: bool = False,
) -> List[AttnLayerCost]:
    """QK / SV / Softmax (or fused) over `kv_tokens` compressed or selected keys."""
    B = dims.batchsize
    H = dims.num_attention_heads
    KV = dims.num_key_value_heads
    D = dims.head_size
    a = dims.a_byte
    kv = dims.kv_byte
    Q = q_tokens
    W = kv_tokens

    qk_ops = Q * W * D * H * B * 2
    sv_ops = Q * D * W * H * B * 2
    softmax_ops = B * H * Q * W * 5

    if use_flashattention:
        block_size_r = min(math.ceil(dims.onchip_buffer / (kv * D)), D)
        n_blocks_r = math.ceil(max(Q, 1) / block_size_r) if block_size_r > 0 else 1
        return [
            AttnLayerCost(
                name="fused_attention",
                OPs=qk_ops + sv_ops + softmax_ops,
                load_act=Q * D * B * H * a,
                store_act=Q * W * B * H * a * 2,
                load_kv_cache=n_blocks_r * W * D * B * KV * kv * 2,
            )
        ]

    return [
        AttnLayerCost(
            name="qk_matmul",
            OPs=qk_ops,
            load_act=W * D * B * KV * a if Q > 1 else 1 * D * B * H * a,
            store_act=Q * W * B * H * a,
            load_kv_cache=W * D * B * KV * kv,
        ),
        AttnLayerCost(
            name="sv_matmul",
            OPs=sv_ops,
            load_act=Q * W * B * H * a,
            store_act=Q * D * B * H * a,
            load_kv_cache=W * D * B * KV * kv,
        ),
        AttnLayerCost(
            name="softmax",
            OPs=softmax_ops,
            load_act=B * H * Q * W * a,
            store_act=B * H * Q * W * a,
        ),
    ]


def compressor_cost(
    dims: AttnDims,
    compress_ratio: int,
    q_tokens: int,
) -> AttnLayerCost:
    """Mean-pool style compressor: raw tokens -> compressed KV tokens.

    Prefill compresses the full sequence once; decode amortizes one new token.
    """
    B = dims.batchsize
    S = dims.seqlen
    KV = dims.num_key_value_heads
    D = dims.head_size
    a = dims.a_byte
    n_comp = n_compressed_tokens(S, compress_ratio)

    if q_tokens <= 1:
        # decode: fold one new token into the compressed stream
        ops = D * KV * B
        load_act = D * B * KV * a
        store_act = D * B * KV * a
    else:
        # prefill: each of S tokens contributes to a pooled slot
        ops = S * D * KV * B
        load_act = S * D * B * KV * a
        store_act = n_comp * D * B * KV * a

    return AttnLayerCost(
        name="compressor",
        OPs=ops,
        load_act=load_act,
        store_act=store_act,
    )


def indexer_cost(
    dims: AttnDims,
    n_comp: int,
    q_tokens: int,
    index_dim: int = 128,
) -> AttnLayerCost:
    """Lightweight CSA indexer: score all compressed positions, keep top-k later."""
    B = dims.batchsize
    a = dims.a_byte
    ops = 2 * n_comp * index_dim * q_tokens * B
    return AttnLayerCost(
        name="indexer",
        OPs=ops,
        load_act=n_comp * index_dim * B * a + q_tokens * index_dim * B * a,
        store_act=q_tokens * n_comp * B * a,  # affinity scores before top-k
    )
