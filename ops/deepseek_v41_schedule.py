"""DeepSeek-V4.1-Flash network schedule.

CED prefill: the 20 encoder layers run on the full prompt; the 20 decoder layers
replay only the last `sliding_window` tokens. Decoder global KV is still built
from the full encoder sequence (one compressor, not 20 decoder forwards).
Decode runs every layer on the new token.

DSpark (speculative head) is not part of this forward.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ops.comm import CommLink, allreduce_bus_bytes, comm_fields, moe_alltoall_bytes
from ops.attention.csa2 import (
    attention_costs,
    compressor_costs,
    index_k_entry_bytes,
    indexer_costs,
    main_kv_entry_bytes,
    n_compressed,
    swa_kv_entry_bytes,
)
from ops.engram import EngramDims, engram_costs, engram_host_bytes
from ops.hc import HcDims, block_hc_costs


def is_v41(model_params) -> bool:
    return getattr(model_params, "model_type", "") == "deepseek_v41"


def _as_list(value) -> List[int]:
    if value is None:
        return []
    return [int(v) for v in value]


def component_order() -> List[str]:
    """Unit-op names emitted for both prefill and decode."""
    names: List[str] = ["hc_expand"]

    def proj(prefix: str) -> None:
        names.extend(
            [
                f"{prefix}_q_a_proj",
                f"{prefix}_q_b_proj",
                f"{prefix}_kv_proj",
                f"{prefix}_o_a_proj",
                f"{prefix}_o_b_proj",
                f"{prefix}_attn_norm",
                f"{prefix}_mlp_norm",
                f"{prefix}_attn_hc_pre",
                f"{prefix}_attn_hc_post",
                f"{prefix}_ffn_hc_pre",
                f"{prefix}_ffn_hc_post",
                f"{prefix}_router",
                f"{prefix}_experts",
            ]
        )

    proj("enc")
    for stem in ("enc_swa", "enc_sparse"):
        names.append(f"{stem}_fused_attention")
    names.extend(["enc_indexer", "enc_compressor"])
    proj("dec")
    for stem in ("dec_swa", "dec_sparse"):
        names.append(f"{stem}_fused_attention")
    names.extend(
        [
            "dec_indexer",
            "dec_reindex_indexer",
            "dec_compressor",
            "engram_lookup",
            "engram_proj",
            "engram_gate",
            "enc_tp_allreduce",
            "dec_tp_allreduce",
            "tp_allreduce_engram",
            "enc_ep_dispatch",
            "enc_ep_combine",
            "dec_ep_dispatch",
            "dec_ep_combine",
            "kv_footprint",
        ]
    )
    return names


def layer_graph(model_params=None) -> Dict[str, List[str]]:
    """Flat dataflow of one V4.1 schedule, for the web viewer."""
    graph: Dict[str, List[str]] = {"input": []}
    prev = "input"
    for name in component_order():
        if name == "kv_footprint":
            continue
        graph[name] = [prev]
        prev = name
    graph["output"] = [prev]
    return graph


def _weight_bytes(n_params: float, bit: int) -> float:
    data = float(n_params) * bit / 8.0
    if bit <= 8:
        data += float(n_params) / 32.0
    return data


def _linear(name: str, ic: int, oc: int, n_tokens: int, bit: int, a_byte: float) -> Dict:
    params = float(ic * oc)
    return {
        "name": name,
        "OPs": 2.0 * ic * oc * n_tokens,
        "load_weight": _weight_bytes(params, bit),
        "load_act": ic * n_tokens * a_byte,
        "store_act": oc * n_tokens * a_byte,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
    }


def _norm(name: str, n_tokens: int, hidden: int, a_byte: float) -> Dict:
    return {
        "name": name,
        "OPs": n_tokens * hidden * 7.0,
        "load_weight": hidden * 4.0,
        "load_act": n_tokens * hidden * a_byte,
        "store_act": n_tokens * hidden * a_byte,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
    }


def _zeros(name: str) -> Dict:
    return {
        "name": name,
        "OPs": 0.0,
        "load_weight": 0.0,
        "load_act": 0.0,
        "store_act": 0.0,
        "load_kv_cache": 0.0,
        "store_kv_cache": 0.0,
    }


def build_network(model_params, ctx: dict) -> Optional[dict]:
    if not is_v41(model_params):
        return None

    batch = int(ctx["batchsize"])
    seqlen = int(ctx["seqlen"])
    a_byte = float(ctx["a_byte"])
    w_bit = int(round(float(ctx["w_byte"]) * 8))
    tp = max(1, int(ctx.get("tp_size") or 1))
    ep = max(1, int(ctx.get("ep_size") or 1))
    link = CommLink(
        bandwidth=float(ctx.get("comm_bandwidth") or 0.0),
        latency=float(ctx.get("comm_latency") or 0.0),
    )

    hidden = int(model_params.hidden_size)
    n_layers = int(model_params.num_hidden_layers)
    n_heads = int(model_params.num_attention_heads)
    head_dim = int(model_params.head_dim)
    q_lora = int(model_params.q_lora_rank)
    o_lora = int(model_params.o_lora_rank)
    o_groups = int(model_params.o_groups)
    window = int(model_params.sliding_window)
    ratios = _as_list(model_params.compress_ratios)
    kv_sources = set(_as_list(model_params.kv_source_layer_ids))
    index_sources = set(_as_list(model_params.index_source_layer_ids))
    index_heads = int(model_params.index_n_heads)
    index_dim = int(model_params.index_head_dim)
    index_topk = int(model_params.index_topk)
    pool = int(model_params.candidate_topk_blocks) * int(model_params.candidate_block_size)
    n_routed = int(model_params.n_routed_experts)
    n_shared = int(model_params.n_shared_experts)
    topk = int(model_params.num_experts_per_tok)
    inter = int(model_params.moe_intermediate_size)
    routed_bit = int(getattr(model_params, "expert_w_bit", 4) or 4)
    hc_mult = int(model_params.hc_mult)
    sinkhorn = int(model_params.hc_sinkhorn_iters)
    n_enc = n_layers // 2
    n_dec = n_layers - n_enc

    if n_heads % tp != 0 or o_groups % tp != 0 or index_heads % tp != 0:
        raise ValueError(
            f"tp_size={tp} must divide n_heads={n_heads}, o_groups={o_groups}, "
            f"index_n_heads={index_heads}"
        )

    q_out = n_heads * head_dim // tp
    o_mid = o_groups * o_lora // tp
    group_in = n_heads * head_dim // o_groups
    swa_only, enc_full, enc_reuse = [], [], []
    dec_full, dec_reindex, dec_reuse = [], [], []
    for i in range(n_layers):
        ratio = ratios[i] if i < len(ratios) else 0
        decoder = i >= n_enc
        if ratio <= 0:
            swa_only.append(i)
        elif i in kv_sources:
            (dec_full if decoder else enc_full).append(i)
        elif i in index_sources:
            (dec_reindex if decoder else enc_reuse).append(i)
        else:
            (dec_reuse if decoder else enc_reuse).append(i)
    if swa_only and any(i >= n_enc for i in swa_only):
        raise ValueError(f"SWA-only layers must sit in the encoder, got {swa_only}")

    n_enc_csa = len(enc_full) + len(enc_reuse)
    enc_ratio = ratios[enc_full[0]] if enc_full else 2
    dec_ratio = ratios[dec_full[0]] if dec_full else 1

    replay = min(seqlen, window)
    swa_kv = min(seqlen, window)
    main_b = main_kv_entry_bytes(head_dim)
    index_b = index_k_entry_bytes(index_dim)
    swa_b = swa_kv_entry_bytes(head_dim)

    def ntok(length: int) -> int:
        return batch * int(length)

    def proj_pair(prefix: str, length_prefill: int):
        specs = [
            ("q_a_proj", hidden, q_lora),
            ("q_b_proj", q_lora, q_out),
            ("kv_proj", hidden, head_dim),
            ("o_a_proj", group_in, o_mid),
            ("o_b_proj", o_mid, hidden),
        ]
        out = []
        for stem, ic, oc in specs:
            out.append(
                (
                    f"{prefix}_{stem}",
                    _linear(stem, ic, oc, ntok(length_prefill), w_bit, a_byte),
                    _linear(stem, ic, oc, ntok(1), w_bit, a_byte),
                )
            )
        return out

    def norm_pair(prefix: str, length_prefill: int):
        out = []
        for stem in ("attn_norm", "mlp_norm"):
            out.append(
                (
                    f"{prefix}_{stem}",
                    _norm(stem, ntok(length_prefill), hidden, a_byte),
                    _norm(stem, ntok(1), hidden, a_byte),
                )
            )
        return out

    def hc_pair(prefix: str, length_prefill: int):
        def one(length: int):
            dims = HcDims(
                batchsize=batch,
                seqlen=int(length),
                hidden_size=hidden,
                hc_mult=hc_mult,
                sinkhorn_iters=sinkhorn,
                w_byte=4.0,  # mHC coefficients are fp32
                a_byte=a_byte,
            )
            return list(block_hc_costs(dims, single_pass=True))

        pre = one(length_prefill)
        dec = one(1)
        out = []
        for p, d in zip(pre, dec):
            out.append(
                (
                    f"{prefix}_{p.name}",
                    {
                        "name": p.name,
                        "OPs": p.OPs,
                        "load_weight": p.load_weight,
                        "load_act": p.load_act,
                        "store_act": p.store_act,
                        "load_kv_cache": p.load_kv_cache,
                        "store_kv_cache": p.store_kv_cache,
                    },
                    {
                        "name": d.name,
                        "OPs": d.OPs,
                        "load_weight": d.load_weight,
                        "load_act": d.load_act,
                        "store_act": d.store_act,
                        "load_kv_cache": d.load_kv_cache,
                        "store_kv_cache": d.store_kv_cache,
                    },
                )
            )
        return out

    if n_routed % ep != 0:
        raise ValueError(f"ep_size={ep} must divide n_routed_experts={n_routed}")
    local_routed = n_routed // ep
    local_topk = topk / ep
    expert_params = 3 * hidden * inter
    # Router stays replicated. Routed experts are sharded; the shared expert is not.
    active_w = local_topk * _weight_bytes(expert_params, routed_bit) + n_shared * _weight_bytes(
        expert_params, w_bit
    )
    full_w = local_routed * _weight_bytes(expert_params, routed_bit) + n_shared * _weight_bytes(
        expert_params, w_bit
    )
    n_active = local_topk + n_shared

    def moe_pair(prefix: str, length_prefill: int):
        def one(length: int, stem: str, ops: float, weight: float, store_h: bool):
            t = ntok(length)
            return {
                "name": stem,
                "OPs": ops * t,
                "load_weight": weight,
                "load_act": hidden * t * a_byte,
                "store_act": (hidden if store_h else n_routed) * t * a_byte,
                "load_kv_cache": 0.0,
                "store_kv_cache": 0.0,
            }

        router_ops = 2.0 * hidden * n_routed
        expert_ops = n_active * (6.0 * hidden * inter + 2.0 * inter)
        router_w = _weight_bytes(hidden * n_routed, w_bit)
        return [
            (
                f"{prefix}_router",
                one(length_prefill, "router", router_ops, router_w, False),
                one(1, "router", router_ops, router_w, False),
            ),
            (
                f"{prefix}_experts",
                one(length_prefill, "experts", expert_ops, active_w, True),
                one(1, "experts", expert_ops, active_w, True),
            ),
        ]

    def add_attn(prefix: str, q_prefill: int, kv: int, entry: float, gather: bool):
        pre = attention_costs(
            prefix, batch, q_prefill, kv, n_heads // tp, head_dim, a_byte, entry, gather
        )
        dec = attention_costs(
            prefix, batch, 1, kv, n_heads // tp, head_dim, a_byte, entry, gather
        )
        return [(p["name"], p, d) for p, d in zip(pre, dec)]

    enc_top = min(index_topk, n_compressed(seqlen, enc_ratio))
    dec_top = min(index_topk, n_compressed(seqlen, dec_ratio))
    enc_scored = n_compressed(seqlen, enc_ratio)
    dec_scored = n_compressed(seqlen, dec_ratio)
    re_scored = min(dec_scored, pool)

    enc_lat_pre = n_compressed(seqlen, enc_ratio)
    enc_lat_dec = 1.0 / enc_ratio
    dec_lat_pre = n_compressed(seqlen, dec_ratio)
    dec_lat_dec = 1.0 / dec_ratio

    def one_indexer(name, q_pre, n_pre, n_dec, lat_pre, lat_dec, owns):
        # Indexer Q is column-parallel; scale its projection by passing heads/tp
        # inside indexer_costs via index_n_heads. K is a single shared head.
        pre = indexer_costs(
            name, batch, q_pre, n_pre, lat_pre, hidden, q_lora, head_dim,
            index_heads // tp, index_dim, owns, a_byte, w_bit / 8.0, index_b,
        )
        dec = indexer_costs(
            name, batch, 1, n_dec, lat_dec, hidden, q_lora, head_dim,
            index_heads // tp, index_dim, owns, a_byte, w_bit / 8.0, index_b,
        )
        # q_lora -> local heads. indexer_costs uses index_n_heads * index_head_dim
        # as the Q width, which we already divided by tp. Good.
        pre["name"] = name
        dec["name"] = name
        return (name, pre, dec)

    # Decoder compressor sees the whole prompt on prefill (KV is projected from
    # encoder states), and one new token on decode.
    enc_comp_pre = compressor_costs(
        "enc_compressor", batch, seqlen, hidden, head_dim, enc_ratio, a_byte, w_bit / 8.0
    )
    enc_comp_dec = compressor_costs(
        "enc_compressor", batch, 1, hidden, head_dim, enc_ratio, a_byte, w_bit / 8.0
    )
    dec_comp_pre = compressor_costs(
        "dec_compressor", batch, seqlen, hidden, head_dim, dec_ratio, a_byte, w_bit / 8.0
    )
    dec_comp_dec = compressor_costs(
        "dec_compressor", batch, 1, hidden, head_dim, dec_ratio, a_byte, w_bit / 8.0
    )

    engram_ids = _as_list(getattr(model_params, "engram_layer_ids", []))
    engram_pre = engram_costs(
        EngramDims(
            batchsize=batch,
            tokens=seqlen,
            hidden_size=hidden,
            hc_mult=hc_mult,
            max_ngram_size=int(model_params.engram_max_ngram_size),
            n_heads=int(model_params.engram_n_heads),
            head_dim=int(model_params.engram_head_dim),
            proj_byte=1.0,
            a_byte=a_byte,
            row_byte=1.0,
        )
    )
    engram_dec = engram_costs(
        EngramDims(
            batchsize=batch,
            tokens=1,
            hidden_size=hidden,
            hc_mult=hc_mult,
            max_ngram_size=int(model_params.engram_max_ngram_size),
            n_heads=int(model_params.engram_n_heads),
            head_dim=int(model_params.engram_head_dim),
            proj_byte=1.0,
            a_byte=a_byte,
            row_byte=1.0,
        )
    )

    groups = []
    groups.append(
        (
            "hc_expand",
            1,
            {
                "name": "hc_expand",
                "OPs": ntok(seqlen) * hc_mult * hidden,
                "load_weight": 0.0,
                "load_act": ntok(seqlen) * hidden * a_byte,
                "store_act": ntok(seqlen) * hc_mult * hidden * a_byte,
                "load_kv_cache": 0.0,
                "store_kv_cache": 0.0,
            },
            {
                "name": "hc_expand",
                "OPs": ntok(1) * hc_mult * hidden,
                "load_weight": 0.0,
                "load_act": ntok(1) * hidden * a_byte,
                "store_act": ntok(1) * hc_mult * hidden * a_byte,
                "load_kv_cache": 0.0,
                "store_kv_cache": 0.0,
            },
        )
    )

    def extend_pairs(pairs, mult: int):
        for name, pre, dec in pairs:
            groups.append((name, mult, pre, dec))

    extend_pairs(proj_pair("enc", seqlen), n_enc)
    extend_pairs(norm_pair("enc", seqlen), n_enc)
    extend_pairs(hc_pair("enc", seqlen), n_enc)
    extend_pairs(moe_pair("enc", seqlen), n_enc)
    extend_pairs(add_attn("enc_swa", seqlen, swa_kv, swa_b, False), n_enc)
    extend_pairs(add_attn("enc_sparse", seqlen, enc_top, main_b, True), n_enc_csa)
    groups.append(
        (
            "enc_indexer",
            len(enc_full),
            *one_indexer(
                "enc_indexer", seqlen, enc_scored, enc_scored, enc_lat_pre, enc_lat_dec, True
            )[1:],
        )
    )
    groups.append(("enc_compressor", len(enc_full), enc_comp_pre, enc_comp_dec))

    extend_pairs(proj_pair("dec", replay), n_dec)
    extend_pairs(norm_pair("dec", replay), n_dec)
    extend_pairs(hc_pair("dec", replay), n_dec)
    extend_pairs(moe_pair("dec", replay), n_dec)
    extend_pairs(add_attn("dec_swa", replay, swa_kv, swa_b, False), n_dec)
    extend_pairs(add_attn("dec_sparse", replay, dec_top, main_b, True), n_dec)
    groups.append(
        (
            "dec_indexer",
            len(dec_full),
            *one_indexer(
                "dec_indexer", replay, dec_scored, dec_scored, dec_lat_pre, dec_lat_dec, True
            )[1:],
        )
    )
    groups.append(
        (
            "dec_reindex_indexer",
            len(dec_reindex),
            *one_indexer(
                "dec_reindex_indexer", replay, re_scored, re_scored, 0, 0, False
            )[1:],
        )
    )
    groups.append(("dec_compressor", len(dec_full), dec_comp_pre, dec_comp_dec))

    for pre_c, dec_c in zip(engram_pre, engram_dec):
        groups.append(
            (
                pre_c.name,
                len(engram_ids),
                {
                    "name": pre_c.name,
                    "OPs": pre_c.OPs,
                    "load_weight": pre_c.load_weight,
                    "load_act": pre_c.load_act,
                    "store_act": pre_c.store_act,
                    "load_kv_cache": 0.0,
                    "store_kv_cache": 0.0,
                },
                {
                    "name": dec_c.name,
                    "OPs": dec_c.OPs,
                    "load_weight": dec_c.load_weight,
                    "load_act": dec_c.load_act,
                    "store_act": dec_c.store_act,
                    "load_kv_cache": 0.0,
                    "store_kv_cache": 0.0,
                },
            )
        )

    def collective_pair(name, length_prefill, per_token_bytes, group, mult):
        def one(length):
            bus = allreduce_bus_bytes(ntok(length) * per_token_bytes, group)
            fields = comm_fields(bus, link, group)
            fields["name"] = name
            return fields

        groups.append((name, mult, one(length_prefill), one(1)))

    # wo_b is row-parallel: each rank holds a partial hidden sum.
    collective_pair("enc_tp_allreduce", seqlen, hidden * a_byte, tp, n_enc)
    collective_pair("dec_tp_allreduce", replay, hidden * a_byte, tp, n_dec)
    engram_width = (
        (int(model_params.engram_max_ngram_size) - 1)
        * int(model_params.engram_n_heads)
        * int(model_params.engram_head_dim)
    )
    collective_pair(
        "tp_allreduce_engram",
        seqlen,
        engram_width * a_byte,
        tp,
        max(1, len(engram_ids)),
    )

    def alltoall_pair(name, length_prefill, group, mult):
        def one(length):
            sent = moe_alltoall_bytes(ntok(length), hidden, a_byte, group, topk)
            fields = comm_fields(sent, link, group)
            fields["name"] = name
            return fields

        groups.append((name, mult, one(length_prefill), one(1)))

    alltoall_pair("enc_ep_dispatch", seqlen, ep, n_enc)
    alltoall_pair("enc_ep_combine", seqlen, ep, n_enc)
    alltoall_pair("dec_ep_dispatch", replay, ep, n_dec)
    alltoall_pair("dec_ep_combine", replay, ep, n_dec)

    entries = 0
    for layer in sorted(kv_sources):
        if layer >= n_layers:
            continue
        entries += n_compressed(seqlen, ratios[layer])
    kv_bytes = entries * (main_b + index_b) + n_layers * swa_kv * swa_b
    groups.append(
        (
            "kv_footprint",
            1,
            {
                "name": "kv_footprint",
                "OPs": 0.0,
                "load_weight": 0.0,
                "load_act": 0.0,
                "store_act": 0.0,
                "load_kv_cache": 0.0,
                "store_kv_cache": kv_bytes,
            },
            _zeros("kv_footprint"),
        )
    )

    multipliers = {}
    layers = []
    for name, mult, pre, dec in groups:
        if name in multipliers:
            raise RuntimeError(f"duplicate schedule op {name}")
        multipliers[name] = int(mult)
        for stage, cost in (("prefill", pre), ("decode", dec)):
            row = dict(cost)
            row["name"] = name
            row["stage"] = stage
            layers.append(row)

    expected = component_order()
    if list(multipliers) != expected:
        missing = [n for n in expected if n not in multipliers]
        extra = [n for n in multipliers if n not in expected]
        raise RuntimeError(f"schedule names drifted, missing={missing} extra={extra}")

    host = engram_host_bytes(
        _as_list(model_params.engram_num_embeddings),
        int(model_params.engram_head_dim),
        1.0,
    )
    note = (
        f"ced=enc{n_enc}*S+dec{n_dec}*min(S,{window}); "
        f"swa_only={swa_only}; full={enc_full + dec_full}; "
        f"reindex={dec_reindex}; reuse={len(enc_reuse)}enc+{len(dec_reuse)}dec; "
        f"engram={engram_ids}; engram_host={host:.3e}B; "
        f"kv=fp4_main({main_b:.0f}B)+fp4_index({index_b:.0f}B)+fp8_swa({swa_b:.0f}B); "
        f"experts=fp{routed_bit}; mhc=single_pass; dspark=off; tp={tp}; ep={ep}"
    )
    return {
        "layers": layers,
        "multipliers": multipliers,
        "weight_footprint_extra": (full_w - active_w) * n_layers,
        "kv_footprint_exact": True,
        "attention_type": "csa2+swa",
        "window_size": window,
        "index_topk": index_topk,
        "note": note,
        "schedule": {
            "n_encoder": n_enc,
            "n_decoder": n_dec,
            "swa_only": swa_only,
            "enc_full": enc_full,
            "enc_reuse": enc_reuse,
            "dec_full": dec_full,
            "dec_reindex": dec_reindex,
            "dec_reuse": dec_reuse,
            "replay": replay,
            "engram_host_bytes": host,
            "kv_bytes": kv_bytes,
        },
    }
