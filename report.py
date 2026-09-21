"""E2E aggregation helpers: category tagging + scaled totals.

This is analysis/reporting, not an operator cost model (those live under ops/).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional


# category -> name matchers (substring / exact). First match wins.
_CATEGORY_RULES = [
    ("moe", ("hash_router", "hash_experts", "router", "experts")),
    (
        "attention",
        (
            "qk_matmul",
            "sv_matmul",
            "softmax",
            "fused_attention",
            "compressor",
            "indexer",
        ),
    ),
    (
        "attn_proj",
        (
            "q_a_proj",
            "q_b_proj",
            "o_a_proj",
            "o_b_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ),
    ),
    ("mlp", ("gate_proj", "up_proj", "down_proj", "mlp_act")),
    ("norm", ("attn_norm", "mlp_norm", "norm")),
    ("residual", ("attn_add", "mlp_add")),
    ("head", ("lm_head",)),
]


def categorize_layer(name: str) -> str:
    for cat, keys in _CATEGORY_RULES:
        if name in keys:
            return cat
        # allow suffix / prefix variants
        for k in keys:
            if name == k or name.endswith("_" + k) or name.startswith(k):
                return cat
    return "other"


def layer_multiplier(
    layer_name: str,
    num_hidden_layers: int,
    num_hash_layers: int = 0,
    num_learned_layers: Optional[int] = None,
) -> int:
    """Match ModelAnalyzer total aggregation weights."""
    if layer_name == "lm_head":
        return 1
    if num_learned_layers is None:
        num_learned_layers = max(0, num_hidden_layers - num_hash_layers)
    if layer_name in ("hash_router", "hash_experts"):
        return max(0, num_hash_layers)
    if layer_name in ("router", "experts"):
        return max(0, num_learned_layers)
    return num_hidden_layers


def aggregate_e2e(
    stage_results: Mapping[str, Mapping[str, float]],
    num_hidden_layers: int,
    num_hash_layers: int = 0,
    num_learned_layers: Optional[int] = None,
    metrics: Iterable[str] = ("inference_time", "OPs", "memory_access"),
) -> Dict:
    """Scale per-block ops to network E2E and bucket by category.

    Returns:
      {
        "e2e": {metric: total, ...},
        "categories": {
           cat: {metric: v, f"{metric}_pct": pct, ...},
           ...
        },
        "scaled_layers": {
           name: {"category", "mult", metric: scaled_v, ...},
        },
      }
    """
    metrics = tuple(metrics)
    if num_learned_layers is None:
        num_learned_layers = max(0, num_hidden_layers - num_hash_layers)

    scaled_layers = {}
    e2e = {m: 0.0 for m in metrics}
    categories = {}

    for name, result in stage_results.items():
        mult = layer_multiplier(
            name, num_hidden_layers, num_hash_layers, num_learned_layers
        )
        cat = categorize_layer(name)
        entry = {"category": cat, "mult": mult}
        if cat not in categories:
            categories[cat] = {m: 0.0 for m in metrics}
        for m in metrics:
            v = float(result.get(m, 0.0) or 0.0) * mult
            entry[m] = v
            e2e[m] += v
            categories[cat][m] += v
        scaled_layers[name] = entry

    # percentages
    for cat, vals in categories.items():
        for m in metrics:
            total = e2e[m]
            vals[f"{m}_pct"] = (100.0 * vals[m] / total) if total > 0 else 0.0

    for name, entry in scaled_layers.items():
        for m in metrics:
            total = e2e[m]
            entry[f"{m}_pct"] = (100.0 * entry[m] / total) if total > 0 else 0.0

    # stable category order
    ordered = {}
    preferred = ["attn_proj", "attention", "moe", "mlp", "norm", "residual", "head", "other"]
    for cat in preferred:
        if cat in categories:
            ordered[cat] = categories[cat]
    for cat, vals in categories.items():
        if cat not in ordered:
            ordered[cat] = vals

    return {"e2e": e2e, "categories": ordered, "scaled_layers": scaled_layers}
