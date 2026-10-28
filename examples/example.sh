
# python analyze_cli.py --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --hardware nvidia_B200

python analyze_cli.py deepseek-ai/DeepSeek-V4-Flash-Vision-Exp nvidia_B200 --vision-tokens 35000

# DeepSeek-V4.1-Flash: ViT + embedding + single-pass mHC + SWA/CSA2 + MoE + Engram
# --w_bit 8 matches the FP8 backbone; routed experts stay FP4 via the model config.
python analyze_cli.py deepseek-ai/DeepSeek-V4.1-Flash nvidia_B200 --w_bit 8 --a_bit 16 --seqlen 4096

# TP shards attention heads, EP shards routed experts, DP splits the global batch.
python analyze_cli.py deepseek-ai/DeepSeek-V4.1-Flash nvidia_B200 --w_bit 8 --a_bit 16 --seqlen 4096 --batchsize 2 --tp-size 8 --ep-size 8 --dp-size 2

