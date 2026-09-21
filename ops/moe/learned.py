"""Learned / dense-gate MoE (DeepSeek-V4 layers after num_hash_layers).

Router is a linear projection hidden -> n_routed, then top-k (+ shared expert).
"""

from __future__ import annotations

from typing import List

from ops.moe.base import BaseMoeOp, MoeDims, MoeLayerCost, experts_cost


class LearnedMoeOp(BaseMoeOp):
    kind = "learned"

    def _router_cost(self, dims: MoeDims, tokens: int) -> MoeLayerCost:
        h, n = dims.hidden_size, dims.n_routed_experts
        return MoeLayerCost(
            name=self._name("router"),
            OPs=2 * h * n * tokens,
            load_weight=h * n * dims.w_byte,
            load_act=h * tokens * dims.a_byte,
            store_act=n * tokens * dims.a_byte,
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
