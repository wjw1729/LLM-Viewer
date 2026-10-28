from model_analyzer import ModelAnalyzer
import numpy as np
import os
import importlib
import argparse
from backend_settings import avaliable_model_ids_sources

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument(
    "hardware",
    type=str,
    help="name of hardware, for example nvidia_V100 or nvidia_A6000",
)
parser.add_argument(
    "--source",
    type=str,
    default=None,
    help="source of model, if not huggingface, will use local model in model_params.<source>",
)
parser.add_argument("--config_file", type=str, default=None, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
parser.add_argument("--promptlen", type=int, default=128, help="prompt sequence length")
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
parser.add_argument(
    "--tp-size",
    type=int,
    default=1,
    help="the number of devices for tensor parallelism to use",
)
parser.add_argument(
    "--ep-size",
    type=int,
    default=1,
    help="expert-parallel group size; routed experts are sharded across the group",
)
parser.add_argument(
    "--attention-type",
    type=str,
    default=None,
    choices=["full", "sliding_window", "swa", "csa", "hca"],
    help="attention cost model: full / sliding_window|swa / csa / hca",
)
parser.add_argument(
    "--window-size",
    type=int,
    default=None,
    help="sliding window size for SWA (implies --attention-type sliding_window if type omitted)",
)
parser.add_argument(
    "--compress-ratio",
    type=int,
    default=None,
    help="KV compress ratio for CSA (default 4) / HCA (default 128)",
)
parser.add_argument(
    "--index-topk",
    type=int,
    default=None,
    help="CSA indexer top-k over compressed tokens (default 512)",
)
parser.add_argument(
    "--vision-tokens",
    type=int,
    default=None,
    help="LLM-side vision tokens after aligner (default: vision_max_n_token if model has ViT; 0 disables)",
)
parser.add_argument(
    "--num-images",
    type=int,
    default=1,
    help="number of images in the prompt (vision prefill only)",
)
args = parser.parse_args()

source = args.source
if source is None:
    source = avaliable_model_ids_sources.get(args.model_id, {}).get("source", "huggingface")

analyzer = ModelAnalyzer(args.model_id, args.hardware, args.config_file, source=source)
ret = analyzer.analyze_generate_task(
    args.promptlen,
    args.seqlen,
    args.batchsize,
    args.w_bit,
    args.a_bit,
    args.kv_bit,
    args.use_flashattention,
    tp_size=args.tp_size,
    ep_size=args.ep_size,
    attention_type=args.attention_type,
    window_size=args.window_size,
    compress_ratio=args.compress_ratio,
    index_topk=args.index_topk,
    vision_tokens=args.vision_tokens,
    num_images=args.num_images,
)
elapse = ret["inference_time"]
prefill_elapse = ret["prefill_time"]
print(
    f"{args.hardware}: 1st token latency {prefill_elapse:.2f}, "
    f"total latency {elapse:.2f}, "
    f"throughput {args.seqlen * args.batchsize / elapse:.2f} Token/sec"
)
