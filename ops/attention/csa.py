"""Compressed Sparse Attention (CSA).

1) Compress KV with compress_ratio (default 4)
2) Indexer selects top-k compressed positions per query (default index_topk=512)
3) Attention runs only over the selected keys

KV footprint still stores the full compressed sequence (n_comp), not just top-k.
"""

from __future__ import annotations

from typing import List, Optional

from ops.attention.base import AttnDims, AttnLayerCost, BaseAttentionOp
from ops.attention.compressed import (
    attention_over_kv,
    compressor_cost,
    indexer_cost,
    n_compressed_tokens,
)


class CompressedSparseAttentionOp(BaseAttentionOp):
    kind = "csa"
    DEFAULT_COMPRESS_RATIO = 4
    DEFAULT_INDEX_TOPK = 512
    DEFAULT_INDEX_DIM = 128

    def __init__(
        self,
        window_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        index_topk: Optional[int] = None,
        index_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(window_size=window_size)
        ratio = self.DEFAULT_COMPRESS_RATIO if compress_ratio is None else int(compress_ratio)
        topk = self.DEFAULT_INDEX_TOPK if index_topk is None else int(index_topk)
        if ratio <= 0:
            raise ValueError(f"CSA compress_ratio must be > 0, got {ratio}")
        if topk <= 0:
            raise ValueError(f"CSA index_topk must be > 0, got {topk}")
        self.compress_ratio = ratio
        self.index_topk = topk
        self.index_dim = self.DEFAULT_INDEX_DIM if index_dim is None else int(index_dim)

    def n_compressed(self, seqlen: int) -> int:
        return n_compressed_tokens(seqlen, self.compress_ratio)

    def effective_kv_len(self, seqlen: int) -> int:
        """Keys actually attended after top-k selection."""
        return min(self.index_topk, self.n_compressed(seqlen))

    def kv_cache_tokens(self, seqlen: int) -> int:
        """Compressed KV cache length (before top-k)."""
        return self.n_compressed(seqlen)

    def decode_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        n_comp = self.n_compressed(dims.seqlen)
        W = self.effective_kv_len(dims.seqlen)
        costs = [
            compressor_cost(dims, self.compress_ratio, q_tokens=1),
            indexer_cost(dims, n_comp=n_comp, q_tokens=1, index_dim=self.index_dim),
        ]
        costs.extend(attention_over_kv(dims, q_tokens=1, kv_tokens=W, use_flashattention=use_flashattention))
        return costs

    def prefill_costs(self, dims: AttnDims, use_flashattention: bool = False) -> List[AttnLayerCost]:
        n_comp = self.n_compressed(dims.seqlen)
        W = self.effective_kv_len(dims.seqlen)
        costs = [
            compressor_cost(dims, self.compress_ratio, q_tokens=dims.seqlen),
            indexer_cost(dims, n_comp=n_comp, q_tokens=dims.seqlen, index_dim=self.index_dim),
        ]
        costs.extend(
            attention_over_kv(
                dims,
                q_tokens=dims.seqlen,
                kv_tokens=W,
                use_flashattention=use_flashattention,
            )
        )
        return costs
