"""Collective communication time (alpha-beta).

Bandwidth is the unidirectional injection rate of one GPU. Algorithm volume:

  allreduce  2*(p-1)/p * message     (ring reduce-scatter + allgather)
  allgather  (p-1)/p * full_tensor    (this rank already holds 1/p)
  alltoall   bytes this rank sends    (NVSwitch / full bisection)

Time is latency + volume / bandwidth, and is not the HBM roofline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommLink:
    bandwidth: float  # bytes/s, unidirectional
    latency: float  # seconds, per collective launch


def group_size(n: int) -> int:
    return max(1, int(n))


def allreduce_bus_bytes(message_bytes: float, group: int) -> float:
    p = group_size(group)
    if p <= 1 or message_bytes <= 0:
        return 0.0
    return 2.0 * (p - 1) / p * float(message_bytes)


def allgather_bus_bytes(full_bytes: float, group: int) -> float:
    """`full_bytes` is the gathered tensor. This rank already owns 1/p of it."""
    p = group_size(group)
    if p <= 1 or full_bytes <= 0:
        return 0.0
    return (p - 1) / p * float(full_bytes)


def alltoall_bus_bytes(sent_bytes: float, group: int) -> float:
    p = group_size(group)
    if p <= 1 or sent_bytes <= 0:
        return 0.0
    return float(sent_bytes)


def comm_time(bus_bytes: float, link: CommLink, group: int) -> float:
    if group_size(group) <= 1 or bus_bytes <= 0 or link.bandwidth <= 0:
        return 0.0
    return float(link.latency) + float(bus_bytes) / float(link.bandwidth)


def moe_remote_sends(ep: int, topk: int) -> float:
    """Expected remote ranks a token is copied to.

    Experts are spread uniformly over `ep` ranks. `topk` draws hit
    `ep * (1 - (1-1/ep)^topk)` ranks, and `(ep-1)/ep` of those are remote.
    """
    p = group_size(ep)
    k = max(0, int(topk))
    if p <= 1 or k <= 0:
        return 0.0
    n_dest = p * (1.0 - (1.0 - 1.0 / p) ** k)
    return n_dest * (p - 1) / p


def moe_alltoall_bytes(
    n_tokens: float,
    hidden: int,
    a_byte: float,
    ep: int,
    topk: int,
) -> float:
    """Dispatch or combine payload: one hidden vector per remote destination."""
    if n_tokens <= 0:
        return 0.0
    return float(n_tokens) * int(hidden) * float(a_byte) * moe_remote_sends(ep, topk)


def comm_fields(bus_bytes: float, link: CommLink, group: int) -> dict:
    """Layer fields for ModelAnalyzer. `comm_time` bypasses the HBM roofline."""
    bus = float(bus_bytes)
    return {
        "OPs": 0.0,
        "load_weight": 0.0,
        "load_act": bus,
        "store_act": 0.0,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
        "comm_time": comm_time(bus, link, group),
    }
