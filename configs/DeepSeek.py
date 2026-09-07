"""DeepSeek-V4 (Flash / Vision-Exp) — Phase 1: text backbone + simplified MoE.

Not modeled yet: CSA/HCA hybrid attention, Hyper-Connections, Vision encoder, MTP/DSpark.
Attention uses a LoRA-style Q/O approximation + MQA KV; FFN is MoE (top-k routed + shared).
"""


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
    """Return MoE routing params, or None if not an MoE model."""
    n_routed = getattr(model_params, "n_routed_experts", None)
    if not n_routed:
        return None
    return {
        "n_routed_experts": int(n_routed),
        "n_shared_experts": int(getattr(model_params, "n_shared_experts", 1)),
        "num_experts_per_tok": int(getattr(model_params, "num_experts_per_tok", 1)),
        "moe_intermediate_size": int(getattr(model_params, "moe_intermediate_size")),
        "hidden_size": int(get_hidden_size(model_params)),
    }


def post_process(model_params, args):
    hiddensize = get_hidden_size(model_params)
    vocab_size = get_vocab_size(model_params)
    layers = []
    for stage in ["prefill", "decode"]:
        layers.append(
            {
                "name": "lm_head",
                "stage": stage,
                "OPs": args["batchsize"] * hiddensize * vocab_size * 1,
                "load_weight": hiddensize * vocab_size * args["w_byte"],
                "load_act": hiddensize * args["a_byte"],
                "store_act": vocab_size * args["a_byte"],
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
    "attn_norm": ["input"],
    "q_a_proj": ["attn_norm"],
    "q_b_proj": ["q_a_proj"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_b_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "o_a_proj": ["sv_matmul"],
    "o_b_proj": ["o_a_proj"],
    "attn_add": ["input", "o_b_proj"],
    "mlp_norm": ["attn_add"],
    "router": ["mlp_norm"],
    "experts": ["mlp_norm", "router"],
    "mlp_add": ["attn_add", "experts"],
    "output": ["mlp_add"],
}

flashattention_transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_a_proj": ["attn_norm"],
    "q_b_proj": ["q_a_proj"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_b_proj", "k_proj", "v_proj"],
    "o_a_proj": ["fused_attention"],
    "o_b_proj": ["o_a_proj"],
    "attn_add": ["input", "o_b_proj"],
    "mlp_norm": ["attn_add"],
    "router": ["mlp_norm"],
    "experts": ["mlp_norm", "router"],
    "mlp_add": ["attn_add", "experts"],
    "output": ["mlp_add"],
}
