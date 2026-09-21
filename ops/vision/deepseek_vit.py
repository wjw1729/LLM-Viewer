"""DeepSeek-ViT + Aligner cost model (DeepSeek-V4-Flash-Vision-Exp).

Reference: inference/vision.py
  - PatchEmbed: Linear(3*P^2 -> D)
  - L x Block: RMSNorm + full attn (wqkv/wo) + RMSNorm + SwiGLU MLP
  - Final RMSNorm
  - Aligner: pixel-unshuffle (R x R) then Linear(D*R^2 -> H) + GELU + Linear(H -> H)
"""

from __future__ import annotations

from typing import List

from ops.vision.base import BaseVisionOp, VisionDims, VisionLayerCost


class DeepSeekViTOp(BaseVisionOp):
    kind = "deepseek_vit"

    def prefill_costs(self, dims: VisionDims) -> List[VisionLayerCost]:
        B = dims.batchsize * dims.num_images
        N = dims.n_vit_tokens
        D = dims.vision_dim
        H = dims.vision_n_heads
        head = D // H
        I = dims.vision_inter_dim
        L = dims.vision_n_layers
        P = dims.vision_patch_size
        R = dims.downsample_ratio
        Hd = dims.llm_hidden_size
        a = dims.a_byte
        w = dims.w_byte
        N_out = dims.n_out_tokens

        costs: List[VisionLayerCost] = []

        # --- patch embed: [3*P^2, D] ---
        patch_in = 3 * P * P
        costs.append(
            VisionLayerCost(
                name="vision_patch_embed",
                OPs=B * N * patch_in * D * 2,
                load_weight=patch_in * D * w,
                load_act=B * N * patch_in * a,
                store_act=B * N * D * a,
            )
        )

        # --- L transformer blocks (aggregated) ---
        # Attention projections: wqkv D->3D, wo D->D
        attn_proj_ops = B * N * (D * 3 * D + D * D) * 2
        attn_proj_w = (D * 3 * D + D * D) * w
        # Full bidirectional attention over N patches
        attn_qk = B * N * N * head * H * 2
        attn_sv = B * N * head * N * H * 2
        attn_sm = B * H * N * N * 5
        attn_act_load = B * N * D * a  # x into attn
        attn_kv_like = B * N * D * a  # materialize q/k/v-ish
        attn_store = B * N * D * a

        costs.append(
            VisionLayerCost(
                name="vision_attn",
                OPs=(attn_proj_ops + attn_qk + attn_sv + attn_sm) * L,
                load_weight=attn_proj_w * L,
                load_act=(attn_act_load + attn_kv_like) * L,
                store_act=attn_store * L,
            )
        )

        # MLP: w1 D->2I (gate+up fused), w2 I->D, + silu*mul
        mlp_ops = B * N * (D * 2 * I + I * D) * 2 + B * N * I * 2
        mlp_w = (D * 2 * I + I * D) * w
        costs.append(
            VisionLayerCost(
                name="vision_mlp",
                OPs=mlp_ops * L,
                load_weight=mlp_w * L,
                load_act=B * N * D * a * L,
                store_act=B * N * D * a * L,
            )
        )

        # RMSNorm: ~7 ops/elem, per block x2 + final (= 2L+1)
        n_norm = 2 * L + 1
        costs.append(
            VisionLayerCost(
                name="vision_norm",
                OPs=B * N * D * 7 * n_norm,
                load_weight=D * w * n_norm,  # norm scales
                load_act=B * N * D * a * n_norm,
                store_act=B * N * D * a * n_norm,
            )
        )

        # --- Aligner: unshuffle packs R^2 spatial tokens into channels, then 2-layer MLP ---
        # rearrange is memory-bound; count element moves as OPs for roofline timing
        costs.append(
            VisionLayerCost(
                name="vision_downsample",
                OPs=B * N * D,
                load_weight=0,
                load_act=B * N * D * a,
                store_act=B * N_out * (D * R * R) * a,
            )
        )
        in_align = D * R * R
        # w1: in_align -> Hd, w2: Hd -> Hd, + gelu
        costs.append(
            VisionLayerCost(
                name="vision_aligner",
                OPs=B * N_out * (in_align * Hd + Hd * Hd) * 2 + B * N_out * Hd,
                load_weight=(in_align * Hd + Hd * Hd) * w,
                load_act=B * N_out * in_align * a,
                store_act=B * N_out * Hd * a,
            )
        )

        return costs
