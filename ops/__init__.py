"""Pluggable operator cost models (attention, MoE, ...)."""

from ops.attention import create_attention_op

__all__ = ["create_attention_op"]
