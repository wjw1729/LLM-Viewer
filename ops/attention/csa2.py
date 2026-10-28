"""CSA2 + SWA attention costs for DeepSeek-V4.1.

Every CSA2 layer attends a local sliding window and `index_topk` compressed
positions. Modes (paper §2.3):

  - swa:    no compressed branch (compress_ratio == 0)
  - full:   this layer writes main KV + indexer K and scores the full context
  - reindex: reuse main KV / indexer K, score a candidate pool with a new query
  - reuse:  reuse both the cache and the latest top-k; no indexer

KV entry sizes follow the V4.1 serving layout (FP4 main KV, FP4 indexer K, FP8 SWA).
"""

from __future__ import annotations

import math
from typing import Dict, List


def n_compressed(seqlen: int, ratio: int) -> int:
    if ratio <= 0:
        return 0
    return max(1, math.ceil(int(seqlen) / int(ratio)))


def fp4_entry_bytes(channels: int, scale_group: int) -> float:
    """E2M1 payload + one scale byte per group."""
    return channels * 0.5 + channels / float(scale_group)


def fp8_entry_bytes(channels: int, scale_group: int = 32) -> float:
    return channels * 1.0 + channels / float(scale_group)


def main_kv_entry_bytes(head_dim: int) -> float:
    # NVFP4-style: E4M3 scale per 16 channels. head_dim=512 -> 288 B.
    return fp4_entry_bytes(head_dim, 16)


def index_k_entry_bytes(index_head_dim: int) -> float:
    # Indexer K is FP4 with a scale per 32 channels. 128 -> 68 B.
    return fp4_entry_bytes(index_head_dim, 32)


def swa_kv_entry_bytes(head_dim: int) -> float:
    # FP8 activation quant, scale per 32. head_dim=512 -> 528 B.
    return fp8_entry_bytes(head_dim, 32)


def _blank(name: str) -> Dict:
    return {
        "name": name,
        "OPs": 0.0,
        "load_weight": 0.0,
        "load_act": 0.0,
        "store_act": 0.0,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
    }


def attention_costs(
    name_prefix: str,
    batch: int,
    q_tokens: int,
    kv_tokens: int,
    n_heads: int,
    head_dim: int,
    a_byte: float,
    kv_entry_bytes: float,
    gather: bool,
) -> List[Dict]:
    """QK + SV + softmax over `kv_tokens` keys.

    gather=True charges one unique key read per query (sparse top-k).
    gather=False charges the key block once (sliding window reused across queries).
    """
    q = max(0, int(q_tokens))
    kv = max(0, int(kv_tokens))
    # sparse_attn is fused: the score tile stays on chip, only Q/O and KV move.
    if q == 0 or kv == 0:
        return [_blank(f"{name_prefix}_fused_attention")]
    b = batch
    h = n_heads
    d = head_dim
    qk_ops = q * kv * d * h * b * 2.0
    sv_ops = q * d * kv * h * b * 2.0
    sm_ops = b * h * q * kv * 5.0
    key_bytes = (q * kv if gather else kv) * kv_entry_bytes * b
    q_bytes = q * d * b * h * a_byte
    return [
        {
            "name": f"{name_prefix}_fused_attention",
            "OPs": qk_ops + sv_ops + sm_ops,
            "load_weight": 0.0,
            "load_act": q_bytes,
            "store_act": q_bytes,
            "load_kv_cache": key_bytes,
            "store_kv_cache": 0.0,
        }
    ]


def indexer_costs(
    name: str,
    batch: int,
    q_tokens: int,
    n_scored: int,
    n_latents: int,
    hidden: int,
    q_lora_rank: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    owns_k: bool,
    a_byte: float,
    w_byte: float,
    index_k_bytes: float,
) -> Dict:
    """Lightning indexer: Q proj + optional K proj + score against compressed K.

    Scores are not materialized; only the key vector and a top-k index row move.
    """
    q = max(0, int(q_tokens))
    n = max(0, int(n_scored))
    b = batch
    q_out = index_n_heads * index_head_dim
    q_proj = 2.0 * q_lora_rank * q_out * q * b
    # per-head combination weights, Linear(hidden -> index_n_heads)
    w_proj = 2.0 * hidden * index_n_heads * q * b
    score = 2.0 * q * n * index_head_dim * index_n_heads * b
    wk = 0.0
    wk_weight = 0.0
    if owns_k and n_latents > 0:
        # n_latents may be fractional on decode (a ratio>1 group completes every R steps).
        wk = 2.0 * float(n_latents) * head_dim * index_head_dim * b
        wk_weight = float(head_dim * index_head_dim) * w_byte
    q_weight = float(q_lora_rank * q_out + hidden * index_n_heads) * w_byte
    if w_byte <= 1.0:
        q_weight += (q_lora_rank * q_out + hidden * index_n_heads) / 32.0
        if owns_k:
            wk_weight += (head_dim * index_head_dim) / 32.0
    return {
        "name": name,
        "OPs": q_proj + w_proj + score + wk,
        "load_weight": q_weight + wk_weight,
        "load_act": q * q_lora_rank * b * a_byte,
        "store_act": q * index_n_heads * b * a_byte,
        "load_kv_cache": n * index_k_bytes * b,
        "store_kv_cache": 0.0,
    }


def compressor_costs(
    name: str,
    batch: int,
    tokens: int,
    hidden: int,
    head_dim: int,
    ratio: int,
    a_byte: float,
    proj_byte: float,
) -> Dict:
    """Learned pool (ratio>1: fp32 wkv+wgate) or plain latent proj (ratio==1)."""
    t = max(0, int(tokens))
    b = batch
    proj = 2.0 * hidden * head_dim * t * b
    params = float(hidden * head_dim)
    if ratio > 1:
        proj *= 2.0  # wkv + wgate
        params *= 2.0
        # ratio>1 compressor weights are fp32 in the reference forward
        w_bytes = params * 4.0
        proj += t * b * head_dim * 5.0  # softmax over the group + weighted sum
    else:
        w_bytes = params * proj_byte
        if proj_byte <= 1.0:
            w_bytes += params / 32.0
        proj += t * b * head_dim * 7.0  # RMSNorm
    return {
        "name": name,
        "OPs": proj,
        "load_weight": w_bytes,
        "load_act": t * b * hidden * a_byte,
        "store_act": t * b * head_dim * a_byte,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
    }
