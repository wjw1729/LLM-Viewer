from easydict import EasyDict

# Values from https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/main/config.json
# Phase 1 uses text/MoE fields only; vision_* kept for later phases.

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
    # Vision (unused in Phase 1)
    vision_n_layers=32,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
)

model_params = {
    "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": _deepseek_v4_flash_vision_exp,
    # Alias for shorter CLI use
    "DeepSeek-V4-Flash-Vision-Exp": _deepseek_v4_flash_vision_exp,
}
