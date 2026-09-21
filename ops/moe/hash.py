"""Hash-table MoE gate (DeepSeek-V4 first num_hash_layers).

Routing uses a frozen hash / lookup table instead of a dense gate matmul.
Expert compute is the same top-k + shared pattern as learned MoE.
"""

from __future__ import annotations

from typing import List

from ops.moe.base import BaseMoeOp, MoeDims, MoeLayerCost, experts_cost


class HashMoeOp(BaseMoeOp):
    kind = "hash"

    def __init__(self, name_prefix: str = "hash_"):
        super().__init__(name_prefix=name_prefix)

    def _router_cost(self, dims: MoeDims, tokens: int) -> MoeLayerCost:
        """Hash lookup: negligible compute vs dense gate; tiny table footprint.

        Model as O(topk) work per token and an int32 routing table of shape
        [n_routed, topk] (or equivalent packed hash buckets).
        """
        topk = dims.num_experts_per_tok
        n = dims.n_routed_experts
        # lookup / gather indices
        ops = topk * tokens
        # frozen hash / routing table (int32)
        table_bytes = n * topk * 4
        return MoeLayerCost(
            name=self._name("router"),
            OPs=ops,
            load_weight=table_bytes,
            load_act=0,  # routing from token-id / hash, not hidden activation matmul
            store_act=topk * tokens * 4,  # selected expert ids
        )

    def decode_costs(self, dims: MoeDims) -> List[MoeLayerCost]:
        tokens = dims.batchsize
        return [
            self._router_cost(dims, tokens),
            experts_cost(dims, tokens, name=self._name("experts")),
        ]

    def prefill_costs(self, dims: MoeDims) -> List[MoeLayerCost]:
        tokens = dims.batchsize * dims.seqlen
        return [
            self._router_cost(dims, tokens),
            experts_cost(dims, tokens, name=self._name("experts")),
        ]
