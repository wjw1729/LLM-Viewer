"""MoE cost-model package (hash gate + learned gate)."""

from __future__ import annotations

from typing import Optional

from ops.moe.base import MoeDims, MoeLayerCost, BaseMoeOp, expert_weight_bytes, n_active_experts
from ops.moe.hash import HashMoeOp
from ops.moe.learned import LearnedMoeOp

__all__ = [
    "MoeDims",
    "MoeLayerCost",
    "BaseMoeOp",
    "HashMoeOp",
    "LearnedMoeOp",
    "create_moe_op",
    "expert_weight_bytes",
    "n_active_experts",
]

_KIND_ALIASES = {
    "learned": "learned",
    "dense_gate": "learned",
    "normal": "learned",
    "hash": "hash",
    "hash_gate": "hash",
}


def create_moe_op(kind: str = "learned", name_prefix: Optional[str] = None) -> BaseMoeOp:
    """Build an MoE cost model.

    kind:
      - "learned" / "dense_gate" / "normal"
      - "hash" / "hash_gate"
    """
    if kind is None:
        kind = "learned"
    key = _KIND_ALIASES.get(str(kind).lower())
    if key is None:
        raise ValueError(
            f"Unknown MoE kind {kind!r}. "
            f"Supported: {sorted(set(_KIND_ALIASES.values()))}"
        )
    if key == "learned":
        prefix = "" if name_prefix is None else name_prefix
        return LearnedMoeOp(name_prefix=prefix)
    if key == "hash":
        prefix = "hash_" if name_prefix is None else name_prefix
        return HashMoeOp(name_prefix=prefix)
    raise ValueError(f"Unhandled MoE kind {key!r}")
