from easydict import EasyDict

# Values from https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/main/config.json

_deepseek_v4_flash_vision_exp = EasyDict(
    architectures=["DeepseekV4ForCausalLM"],
    model_type="deepseek_v4",
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=1,
    num_hidden_layers=43,
    head_dim=512,
    vocab_size=129280,
    q_lora_rank=1024,
    o_lora_rank=1024,
    qk_rope_head_dim=64,
    # MoE
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=6,
    moe_intermediate_size=2048,
    num_hash_layers=3,  # first 3 layers: hash MoE; rest: learned gate
    # Manifold-Constrained Hyper-Connections (mHC)
    hc_mult=4,
    hc_sinkhorn_iters=20,
    # Vision encoder (DeepSeek-ViT + Aligner)
    vision_n_layers=32,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_patch_size=14,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    vision_max_n_token=384,
    vision_min_pixels=147456,
    vision_max_wh_ratio=8,
)

# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json
# Backbone is FP8; routed experts are FP4. Engram tables are FP8 and host-resident.
_deepseek_v41_flash = EasyDict(
    architectures=["DeepseekV41ForCausalLM"],
    model_type="deepseek_v41",
    hidden_size=5120,
    num_attention_heads=64,
    num_key_value_heads=1,
    num_hidden_layers=40,
    head_dim=512,
    vocab_size=129280,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    qk_rope_head_dim=64,
    # MoE: learned gate on every layer (no hash layers). Routed experts are FP4.
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    moe_intermediate_size=2304,
    num_hash_layers=0,
    expert_w_bit=4,
    # CSA2 + SWA. compress_ratios covers backbone (40) + DSpark draft blocks (3).
    sliding_window=128,
    # 2 SWA + 18 encoder CSA2(ratio=2) + 20 decoder CSA2(ratio=1) + 3 DSpark blocks
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    index_n_heads=32,
    index_head_dim=128,
    index_topk=512,
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    # Single-Pass mHC
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_single_pass=True,
    # Engram modules at layers 1 and 14. num_embeddings is the row count of each table.
    engram_layer_ids=[1, 14],
    engram_num_embeddings=[384006168, 384016682],
    engram_max_ngram_size=4,
    engram_vocab_size=16000000,
    engram_n_heads=8,
    engram_head_dim=256,
    # Vision encoder (DeepSeek-ViT + 3x3 unshuffle + projector)
    vision_n_layers=32,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_patch_size=14,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    vision_max_n_token=1024,
    vision_min_pixels=295936,
    vision_max_wh_ratio=None,
)

model_params = {
    "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": _deepseek_v4_flash_vision_exp,
    # Alias for shorter CLI use
    "DeepSeek-V4-Flash-Vision-Exp": _deepseek_v4_flash_vision_exp,
    "deepseek-ai/DeepSeek-V4.1-Flash": _deepseek_v41_flash,
    "DeepSeek-V4.1-Flash": _deepseek_v41_flash,
}
