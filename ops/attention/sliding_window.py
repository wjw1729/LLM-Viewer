"""Sliding Window Attention (SWA) cost model.

Each query attends to at most `window_size` keys (causal local window).
KV cache footprint is also capped at `window_size` tokens.
"""

from __future__ import annotations

import math
from typing import List, Optional

from ops.attention.base import AttnDims, AttnLayerCost, BaseAttentionOp


class SlidingWindowAttentionOp(BaseAttentionOp):
    kind = "sliding_window"

    def __init__(self, window_size: Optional[int] = None):
        if window_size is None or window_size <= 0:
            raise ValueError(
                "SlidingWindowAttentionOp requires window_size > 0 "
                "(pass --window-size or config.get_sliding_window_size)."
            )
        super().__init__(window_size=int(window_size))

    def effective_kv_len(self, seqlen: int) -> int:
        return min(seqlen, self.window_size)

    def kv_cache_tokens(self, seqlen: int) -> int:
        return self.effective_kv_len(seqlen)

    def decode_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        B = dims.batchsize
        W = self.effective_kv_len(dims.seqlen)
        H = dims.num_attention_heads
        KV = dims.num_key_value_heads
        D = dims.head_size
        a = dims.a_byte
        kv = dims.kv_byte

        # Decode: one query attends to the last W cached keys/values
        qk_ops = W * D * H * B * 2
        sv_ops = 1 * D * W * H * B * 2
        softmax_ops = B * H * W * 1 * 5

        if use_flashattention:
            block_size_r = min(math.ceil(dims.onchip_buffer / (kv * D)), D)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = 1 * D * B * H * a
            o_numel = 1 * W * B * H * a
            return [
                AttnLayerCost(
                    name="fused_attention",
                    OPs=qk_ops + sv_ops + softmax_ops,
                    load_act=q_numel,
                    store_act=o_numel * 2,
                    load_kv_cache=n_blocks_r * W * D * B * KV * kv * 2,
                )
            ]

        return [
            AttnLayerCost(
                name="qk_matmul",
                OPs=qk_ops,
                load_act=1 * D * B * H * a,
                store_act=1 * W * B * H * a,
                load_kv_cache=W * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="sv_matmul",
                OPs=sv_ops,
                load_act=1 * W * B * H * a,
                store_act=1 * D * B * H * a,
                load_kv_cache=W * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="softmax",
                OPs=softmax_ops,
                load_act=B * H * W * 1 * a,
                store_act=B * H * W * 1 * a,
            ),
        ]

    def prefill_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        B = dims.batchsize
        S = dims.seqlen
        W = self.effective_kv_len(S)
        H = dims.num_attention_heads
        KV = dims.num_key_value_heads
        D = dims.head_size
        a = dims.a_byte
        kv = dims.kv_byte

        # Prefill: S queries, each attending ~W keys (same S*K style as full attn)
        qk_ops = S * W * D * H * B * 2
        sv_ops = S * D * W * H * B * 2
        softmax_ops = B * H * S * W * 5

        if use_flashattention:
            block_size_r = min(math.ceil(dims.onchip_buffer / (kv * D)), D)
            n_blocks_r = math.ceil(W / block_size_r) if block_size_r > 0 else 1
            q_numel = S * D * B * H * a
            o_numel = S * W * B * H * a
            return [
                AttnLayerCost(
                    name="fused_attention",
                    OPs=qk_ops + sv_ops + softmax_ops,
                    load_act=q_numel,
                    store_act=o_numel * 2,
                    load_kv_cache=n_blocks_r * W * D * B * KV * kv * 2,
                )
            ]

        return [
            AttnLayerCost(
                name="qk_matmul",
                OPs=qk_ops,
                load_act=W * D * B * KV * a,
                store_act=S * W * B * H * a,
                load_kv_cache=W * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="sv_matmul",
                OPs=sv_ops,
                load_act=S * W * B * H * a,
                store_act=S * D * B * H * a,
                load_kv_cache=W * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="softmax",
                OPs=softmax_ops,
                load_act=B * H * S * W * a,
                store_act=B * H * S * W * a,
            ),
        ]
