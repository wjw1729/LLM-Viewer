"""Heavily Compressed Attention (HCA).

KV is pooled with non-overlap compress_ratio (default 128), then dense attention
runs over the compressed tokens only. No indexer (unlike CSA).
"""

from __future__ import annotations

from typing import List, Optional

from ops.attention.base import AttnDims, AttnLayerCost, BaseAttentionOp
from ops.attention.compressed import (
    attention_over_kv,
    compressor_cost,
    n_compressed_tokens,
)


class HeavilyCompressedAttentionOp(BaseAttentionOp):
    kind = "hca"
    DEFAULT_COMPRESS_RATIO = 128

    def __init__(
        self,
        window_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        index_topk: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(window_size=window_size)
        ratio = self.DEFAULT_COMPRESS_RATIO if compress_ratio is None else int(compress_ratio)
        if ratio <= 0:
            raise ValueError(f"HCA compress_ratio must be > 0, got {ratio}")
        self.compress_ratio = ratio

    def n_compressed(self, seqlen: int) -> int:
        return n_compressed_tokens(seqlen, self.compress_ratio)

    def effective_kv_len(self, seqlen: int) -> int:
        return self.n_compressed(seqlen)

    def kv_cache_tokens(self, seqlen: int) -> int:
        return self.n_compressed(seqlen)

    def decode_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        W = self.effective_kv_len(dims.seqlen)
        costs = [compressor_cost(dims, self.compress_ratio, q_tokens=1)]
        costs.extend(attention_over_kv(dims, q_tokens=1, kv_tokens=W, use_flashattention=use_flashattention))
        return costs

    def prefill_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        W = self.effective_kv_len(dims.seqlen)
        costs = [compressor_cost(dims, self.compress_ratio, q_tokens=dims.seqlen)]
        costs.extend(
            attention_over_kv(
                dims,
                q_tokens=dims.seqlen,
                kv_tokens=W,
                use_flashattention=use_flashattention,
            )
        )
        return costs
