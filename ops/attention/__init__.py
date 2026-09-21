"""Attention cost-model package.

Factory + shared types. Concrete ops live in sibling modules so CSA/HCA/etc.
can be added later without touching ModelAnalyzer.
"""

from __future__ import annotations

from typing import Optional

from ops.attention.base import AttnDims, AttnLayerCost, BaseAttentionOp
from ops.attention.full import FullAttentionOp
from ops.attention.sliding_window import SlidingWindowAttentionOp

__all__ = [
    "AttnDims",
    "AttnLayerCost",
    "BaseAttentionOp",
    "FullAttentionOp",
    "SlidingWindowAttentionOp",
    "create_attention_op",
]

_KIND_ALIASES = {
    "full": "full",
    "dense": "full",
    "sliding_window": "sliding_window",
    "swa": "sliding_window",
}


def create_attention_op(kind: str = "full", window_size: Optional[int] = None) -> BaseAttentionOp:
    """Build an attention cost model by name.

    kind:
      - "full" / "dense"
      - "sliding_window" / "swa"
    """
    if kind is None:
        kind = "full"
    key = _KIND_ALIASES.get(str(kind).lower())
    if key is None:
        raise ValueError(
            f"Unknown attention kind {kind!r}. "
            f"Supported: {sorted(set(_KIND_ALIASES.values()))} (aliases: {sorted(_KIND_ALIASES)})"
        )
    if key == "full":
        return FullAttentionOp(window_size=window_size)
    if key == "sliding_window":
        return SlidingWindowAttentionOp(window_size=window_size)
    raise ValueError(f"Unhandled attention kind {key!r}")
