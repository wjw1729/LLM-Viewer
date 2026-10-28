"""DeepSeek-V4 and V4.1 cost wiring.

V4-Flash-Vision-Exp: LoRA Q/O + MQA, hash MoE then learned MoE, optional ViT,
multi-pass mHC. One attention op is applied to every layer.

V4.1-Flash: CED (20 encoder + 20 decoder), SWA on every layer, CSA2 on the rest,
learned MoE only, Single-Pass mHC, Engram at layers 1 and 14, DeepSeek-ViT.
That schedule is built by ops.deepseek_v41_schedule and is not the uniform
per-layer loop. DSpark is omitted.
"""

from ops.hc import HcDims, embedding_cost, hc_head_cost
from ops.vision import VisionDims, create_vision_op, vision_token_geometry


def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")


def get_head_size(model_params):
    if hasattr(model_params, "head_dim"):
        return getattr(model_params, "head_dim")
    return get_hidden_size(model_params) // get_num_attention_heads(model_params)


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    return getattr(model_params, "moe_intermediate_size")


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def get_moe_info(model_params):
    """Return MoE routing params, or None if not an MoE model.

    DeepSeek-V4: first `num_hash_layers` use hash-table gates; the rest use
    a learned dense gate (top-k + shared expert).
    """
    n_routed = getattr(model_params, "n_routed_experts", None)
    if not n_routed:
        return None
    num_layers = int(get_num_hidden_layers(model_params))
    num_hash_layers = int(getattr(model_params, "num_hash_layers", 0) or 0)
    num_hash_layers = max(0, min(num_hash_layers, num_layers))
    return {
        "n_routed_experts": int(n_routed),
        "n_shared_experts": int(getattr(model_params, "n_shared_experts", 1)),
        "num_experts_per_tok": int(getattr(model_params, "num_experts_per_tok", 1)),
        "moe_intermediate_size": int(getattr(model_params, "moe_intermediate_size")),
        "hidden_size": int(get_hidden_size(model_params)),
        "num_hash_layers": num_hash_layers,
        "num_learned_layers": num_layers - num_hash_layers,
    }


def get_hc_info(model_params):
    """mHC params; None if Hyper-Connections are disabled."""
    hc_mult = int(getattr(model_params, "hc_mult", 0) or 0)
    if hc_mult <= 1:
        return None
    return {
        "hc_mult": hc_mult,
        "hc_sinkhorn_iters": int(getattr(model_params, "hc_sinkhorn_iters", 20) or 20),
        "hidden_size": int(get_hidden_size(model_params)),
    }


def get_vision_info(model_params):
    """Vision encoder params; None if text-only checkpoint."""
    n_layers = int(getattr(model_params, "vision_n_layers", 0) or 0)
    if n_layers <= 0:
        return None
    return {
        "vision_n_layers": n_layers,
        "vision_dim": int(getattr(model_params, "vision_dim")),
        "vision_n_heads": int(getattr(model_params, "vision_n_heads")),
        "vision_inter_dim": int(getattr(model_params, "vision_inter_dim")),
        "vision_patch_size": int(getattr(model_params, "vision_patch_size", 14)),
        "vision_downsample_ratio": int(getattr(model_params, "vision_downsample_ratio", 3)),
        "vision_max_n_token": int(getattr(model_params, "vision_max_n_token", 384)),
        "llm_hidden_size": int(get_hidden_size(model_params)),
        "kind": "deepseek_vit",
    }


def pre_process(model_params, args):
    """Embedding (both stages) + optional vision encoder/aligner (prefill)."""
    layers = []
    batchsize = int(args["batchsize"])
    seqlen = int(args.get("seqlen", 1) or 1)
    w_byte = float(args["w_byte"])
    a_byte = float(args["a_byte"])
    hidden_size = int(get_hidden_size(model_params))
    vocab_size = int(get_vocab_size(model_params))

    # Token embedding: prefill uses full seq; decode embeds one new token.
    layers.append(
        embedding_cost(
            batchsize, seqlen, hidden_size, vocab_size, w_byte, a_byte
        ).as_kwargs("prefill")
    )
    layers.append(
        embedding_cost(
            batchsize, 1, hidden_size, vocab_size, w_byte, a_byte
        ).as_kwargs("decode")
    )

    vision = get_vision_info(model_params)
    if vision is None:
        return layers

    n_out = int(args.get("vision_tokens", 0) or 0)
    if n_out <= 0:
        return layers

    num_images = max(1, int(args.get("num_images", 1) or 1))
    n_out = min(n_out, vision["vision_max_n_token"])
    n_vit, n_out = vision_token_geometry(n_out, vision["vision_downsample_ratio"])

    dims = VisionDims(
        batchsize=batchsize,
        num_images=num_images,
        n_vit_tokens=n_vit,
        n_out_tokens=n_out,
        vision_dim=vision["vision_dim"],
        vision_n_heads=vision["vision_n_heads"],
        vision_inter_dim=vision["vision_inter_dim"],
        vision_n_layers=vision["vision_n_layers"],
        vision_patch_size=vision["vision_patch_size"],
        downsample_ratio=vision["vision_downsample_ratio"],
        llm_hidden_size=vision["llm_hidden_size"],
        w_byte=w_byte,
        a_byte=a_byte,
    )
    op = create_vision_op(vision.get("kind", "deepseek_vit"))
    for cost in op.prefill_costs(dims):
        layers.append(cost.as_kwargs("prefill"))
    return layers


def post_process(model_params, args):
    hiddensize = get_hidden_size(model_params)
    vocab_size = get_vocab_size(model_params)
    batchsize = int(args["batchsize"])
    seqlen = int(args.get("seqlen", 1) or 1)
    w_byte = float(args["w_byte"])
    a_byte = float(args["a_byte"])
    layers = []

    hc = get_hc_info(model_params)
    if hc is not None:
        # V4.1 CED only materializes decoder states for the SWA replay window.
        hc_prefill_len = seqlen
        if getattr(model_params, "model_type", "") == "deepseek_v41":
            win = int(getattr(model_params, "sliding_window", 128) or 128)
            hc_prefill_len = min(seqlen, win)
        for stage, s in (("prefill", hc_prefill_len), ("decode", 1)):
            dims = HcDims(
                batchsize=batchsize,
                seqlen=s,
                hidden_size=int(hc["hidden_size"]),
                hc_mult=int(hc["hc_mult"]),
                sinkhorn_iters=int(hc["hc_sinkhorn_iters"]),
                w_byte=w_byte,
                a_byte=a_byte,
            )
            layers.append(hc_head_cost(dims).as_kwargs(stage))

    for stage in ["prefill", "decode"]:
        layers.append(
            {
                "name": "lm_head",
                "stage": stage,
                "OPs": batchsize * hiddensize * vocab_size * 1,
                "load_weight": hiddensize * vocab_size * w_byte,
                "load_act": hiddensize * a_byte,
                "store_act": vocab_size * a_byte,
            }
        )
    return layers


def get_linear_layers(model_params, tp_size: int):
    """Attention projections only; MoE FFN is handled via get_moe_info."""
    hidden_size = get_hidden_size(model_params)
    head_size = get_head_size(model_params)
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)
    q_lora_rank = int(getattr(model_params, "q_lora_rank", hidden_size))
    o_lora_rank = int(getattr(model_params, "o_lora_rank", hidden_size))

    q_out = attention_heads * head_size
    kv_out = key_value_heads * head_size

    if tp_size > 1:
        assert q_out % tp_size == 0
        assert kv_out % tp_size == 0
        assert q_lora_rank % tp_size == 0 or tp_size == 1

    return {
        # LoRA-style Q: hidden -> q_lora -> heads*head_dim
        "q_a_proj": [hidden_size, q_lora_rank // tp_size],
        "q_b_proj": [q_lora_rank, q_out // tp_size],
        "k_proj": [hidden_size, kv_out // tp_size],
        "v_proj": [hidden_size, kv_out // tp_size],
        # LoRA-style O: heads*head_dim -> o_lora -> hidden
        "o_a_proj": [q_out // tp_size, o_lora_rank],
        "o_b_proj": [o_lora_rank // tp_size, hidden_size],
    }


# name -> input_names
transformer_layer_graph = {
    "input": [],
    "attn_hc_pre": ["input"],
    "attn_norm": ["attn_hc_pre"],
    "q_a_proj": ["attn_norm"],
    "q_b_proj": ["q_a_proj"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_b_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "o_a_proj": ["sv_matmul"],
    "o_b_proj": ["o_a_proj"],
    "attn_hc_post": ["input", "o_b_proj"],
    "ffn_hc_pre": ["attn_hc_post"],
    "mlp_norm": ["ffn_hc_pre"],
    "router": ["mlp_norm"],
    "experts": ["mlp_norm", "router"],
    "ffn_hc_post": ["attn_hc_post", "experts"],
    "output": ["ffn_hc_post"],
}

def network_costs(model_params, ctx):
    """Per-op costs for models that are not a repeated homogeneous block.

    Returns None for V4-Flash, which keeps the uniform layer loop.
    """
    from ops.deepseek_v41_schedule import build_network

    return build_network(model_params, ctx)


def get_transformer_layer_graph(model_params, use_flashattention=False):
    from ops.deepseek_v41_schedule import is_v41, layer_graph

    if is_v41(model_params):
        return layer_graph(model_params)
    if use_flashattention:
        return flashattention_transformer_layer_graph
    return transformer_layer_graph


flashattention_transformer_layer_graph = {
    "input": [],
    "attn_hc_pre": ["input"],
    "attn_norm": ["attn_hc_pre"],
    "q_a_proj": ["attn_norm"],
    "q_b_proj": ["q_a_proj"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_b_proj", "k_proj", "v_proj"],
    "o_a_proj": ["fused_attention"],
    "o_b_proj": ["o_a_proj"],
    "attn_hc_post": ["input", "o_b_proj"],
    "ffn_hc_pre": ["attn_hc_post"],
    "mlp_norm": ["ffn_hc_pre"],
    "router": ["mlp_norm"],
    "experts": ["mlp_norm", "router"],
    "ffn_hc_post": ["attn_hc_post", "experts"],
    "output": ["ffn_hc_post"],
}
