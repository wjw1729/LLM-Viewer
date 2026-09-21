"""Full (dense) attention cost model — matches the original ModelAnalyzer formulas."""

from __future__ import annotations

import math
from typing import List, Optional

from ops.attention.base import AttnDims, AttnLayerCost, BaseAttentionOp


class FullAttentionOp(BaseAttentionOp):
    kind = "full"

    def __init__(self, window_size: Optional[int] = None):
        super().__init__(window_size=window_size)

    def effective_kv_len(self, seqlen: int) -> int:
        return seqlen

    def kv_cache_tokens(self, seqlen: int) -> int:
        return seqlen

    def decode_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        B = dims.batchsize
        S = dims.seqlen
        H = dims.num_attention_heads
        KV = dims.num_key_value_heads
        D = dims.head_size
        a = dims.a_byte
        kv = dims.kv_byte

        qk_ops = S * D * H * B * 2
        sv_ops = 1 * D * S * H * B * 2
        softmax_ops = B * H * S * 1 * 5

        if use_flashattention:
            block_size_r = min(math.ceil(dims.onchip_buffer / (kv * D)), D)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = 1 * D * B * H * a
            o_numel = 1 * S * B * H * a
            return [
                AttnLayerCost(
                    name="fused_attention",
                    OPs=qk_ops + sv_ops + softmax_ops,
                    load_act=q_numel,
                    store_act=o_numel * 2,
                    load_kv_cache=n_blocks_r * S * D * B * KV * kv * 2,
                )
            ]

        return [
            AttnLayerCost(
                name="qk_matmul",
                OPs=qk_ops,
                load_act=1 * D * B * H * a,
                store_act=1 * S * B * H * a,
                load_kv_cache=S * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="sv_matmul",
                OPs=sv_ops,
                load_act=1 * S * B * H * a,
                store_act=1 * D * B * H * a,
                load_kv_cache=S * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="softmax",
                OPs=softmax_ops,
                load_act=B * H * S * 1 * a,
                store_act=B * H * S * 1 * a,
            ),
        ]

    def prefill_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        B = dims.batchsize
        S = dims.seqlen
        H = dims.num_attention_heads
        KV = dims.num_key_value_heads
        D = dims.head_size
        a = dims.a_byte
        kv = dims.kv_byte

        qk_ops = S * S * D * H * B * 2
        sv_ops = S * D * S * H * B * 2
        softmax_ops = B * H * S * S * 5

        if use_flashattention:
            block_size_r = min(math.ceil(dims.onchip_buffer / (kv * D)), D)
            n_blocks_r = math.ceil(S / block_size_r)
            q_numel = S * D * B * H * a
            o_numel = S * S * B * H * a
            return [
                AttnLayerCost(
                    name="fused_attention",
                    OPs=qk_ops + sv_ops + softmax_ops,
                    load_act=q_numel,
                    store_act=o_numel * 2,
                    load_kv_cache=n_blocks_r * S * D * B * KV * kv * 2,
                )
            ]

        return [
            AttnLayerCost(
                name="qk_matmul",
                OPs=qk_ops,
                load_act=S * D * B * KV * a,
                store_act=S * S * B * H * a,
                load_kv_cache=S * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="sv_matmul",
                OPs=sv_ops,
                load_act=S * S * B * H * a,
                store_act=S * D * B * H * a,
                load_kv_cache=S * D * B * KV * kv,
            ),
            AttnLayerCost(
                name="softmax",
                OPs=softmax_ops,
                load_act=B * H * S * S * a,
                store_act=B * H * S * S * a,
            ),
        ]
