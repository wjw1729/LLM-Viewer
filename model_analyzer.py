import os
import importlib
from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import str_number, str_number_time
import math
from ops.attention import AttnDims, create_attention_op
from ops.moe import MoeDims, create_moe_op
from ops.hc import HcDims, block_hc_costs
from report import aggregate_e2e

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]


class ModelAnalyzer:
    def __init__(self, model_id, hardware, config_file=None, source="huggingface"):
        """
        source: 'huggingface' or 'DiT'
        """
        self.model_id = model_id
        self.hardware = hardware
        if config_file is None:
            # get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # auto search the config
            for file in os.listdir(current_dir + "/configs"):
                if file.endswith(".py") and file.replace(".py", "") in model_id:
                    config_file = "configs/" + file
                # print(f"auto search config file {config_file} {file} {model_id}")
        assert config_file is not None, "config file is not found, please specify it manually."
        print(f"use config file {config_file} for {model_id}")
        if source == "huggingface":
            self.model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            if not os.path.exists(f"model_params/{source}.py"):
                raise Exception(f"model_params/{source}.py is not found")
            # from model_params.DiT import model_params
            module = importlib.import_module(f"model_params.{source}")
            self.model_params = module.model_params[model_id]
        self.config = importlib.import_module(config_file.replace("/", ".").replace(".py", ""))

        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None
        self.attention_type = None
        self.window_size = None
        self.compress_ratio = None
        self.index_topk = None

    def _resolve_attention_type(
        self,
        attention_type=None,
        window_size=None,
        compress_ratio=None,
        index_topk=None,
    ):
        cfg = self.config
        if attention_type is None and hasattr(cfg, "get_attention_type"):
            attention_type = cfg.get_attention_type(self.model_params)
        if window_size is None and hasattr(cfg, "get_sliding_window_size"):
            window_size = cfg.get_sliding_window_size(self.model_params)
        if compress_ratio is None and hasattr(cfg, "get_compress_ratio"):
            compress_ratio = cfg.get_compress_ratio(self.model_params)
        if index_topk is None and hasattr(cfg, "get_index_topk"):
            index_topk = cfg.get_index_topk(self.model_params)

        if attention_type is None:
            if window_size is not None and window_size > 0:
                attention_type = "sliding_window"
            else:
                attention_type = "full"

        self.attention_type = attention_type
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.index_topk = index_topk
        return create_attention_op(
            kind=attention_type,
            window_size=window_size,
            compress_ratio=compress_ratio,
            index_topk=index_topk,
        )

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
        # Pure memory moves (OPs==0) are timed as memory_access / bandwidth.
        if performance == 0:
            inference_time = (memory_access / bandwidth) if bandwidth > 0 else 0.0
        else:
            inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }

    def save_csv(self, save_path=None):
        if save_path is None:
            save_path = f"output/{self.model_id[:self.model_id.rfind('/')]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path += f"{self.model_id[self.model_id.rfind('/'):]}"

        decode_file_name = f"{save_path}_decode.csv"
        prefill_file_name = f"{save_path}_prefill.csv"
        print(f"save to {decode_file_name} and {prefill_file_name}")

        num_hidden_layers = self.config.get_num_hidden_layers(self.model_params)
        moe_info = (
            self.config.get_moe_info(self.model_params)
            if hasattr(self.config, "get_moe_info")
            else None
        )
        n_hash = int((moe_info or {}).get("num_hash_layers", 0) or 0)
        n_learned = int(
            (moe_info or {}).get("num_learned_layers", num_hidden_layers - n_hash)
        )

        attn_meta = f"attn={self.attention_type}"
        if self.window_size:
            attn_meta += f",window={self.window_size}"
        if self.compress_ratio:
            attn_meta += f",compress_ratio={self.compress_ratio}"
        if self.index_topk:
            attn_meta += f",index_topk={self.index_topk}"

        for file_name, stage in [
            (decode_file_name, "decode"),
            (prefill_file_name, "prefill"),
        ]:
            stage_label = "decode_step" if stage == "decode" else "prefill"
            report = aggregate_e2e(
                self.results[stage],
                num_hidden_layers=num_hidden_layers,
                num_hash_layers=n_hash,
                num_learned_layers=n_learned,
            )
            total = self.results.get("total_results", {}).get(stage, {})
            e2e_time = report["e2e"]["inference_time"]
            # Prefer analyzer total_results when present (includes footprint fields)
            if total.get("inference_time") is not None:
                # category sum should match; keep report e2e for category consistency
                pass

            with open(file_name, "a+") as f:
                f.write(
                    f"\n\n=== {self.model_id} {self.hardware} w_bit={self.w_bit} "
                    f"a_bit={self.a_bit} kv_bit={self.kv_bit} batchsize={self.batchsize} "
                    f"seqlen={self.seqlen} tp_size={self.tp_size} {attn_meta} ===\n"
                )
                f.write(
                    "# Per-block ops (one transformer block; MoE shows both hash & learned routers)\n"
                )
                f.write(
                    "layer_name,OPs,Access,arithmetic_intensity,performance,bound,"
                    "load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
                )
                for layer_name, result in self.results[stage].items():
                    f.write(
                        f"{layer_name},{str_number(result['OPs'])},"
                        f"{str_number(result['memory_access'])}B,"
                        f"{str_number(result['arithmetic_intensity'])},"
                        f"{str_number(result['performance'])},"
                        f"{result['bound']},"
                        f"{str_number(result['load_weight'])}B,"
                        f"{str_number(result['load_act'])}B,"
                        f"{str_number(result['store_act'])}B,"
                        f"{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,"
                        f"{str_number_time(result['inference_time'])}s\n"
                    )

                # --- E2E summary ---
                f.write(f"\n=== E2E Summary ({stage_label}) ===\n")
                f.write("metric,value\n")
                f.write(f"e2e_inference_time_s,{e2e_time}\n")
                f.write(f"e2e_inference_time,{str_number_time(e2e_time)}s\n")
                f.write(f"e2e_OPs,{report['e2e']['OPs']}\n")
                f.write(f"e2e_OPs_str,{str_number(report['e2e']['OPs'])}\n")
                f.write(f"e2e_memory_access_B,{report['e2e']['memory_access']}\n")
                f.write(
                    f"e2e_memory_access,{str_number(report['e2e']['memory_access'])}B\n"
                )
                if total:
                    if "memory_consumption" in total:
                        f.write(
                            f"memory_consumption_B,{total['memory_consumption']}\n"
                        )
                        f.write(
                            f"memory_consumption,{str_number(total['memory_consumption'])}B\n"
                        )
                    if "memory_consumption_weight" in total:
                        f.write(
                            f"memory_consumption_weight_B,{total['memory_consumption_weight']}\n"
                        )
                    if "memory_consumption_kv_cache" in total:
                        f.write(
                            f"memory_consumption_kv_cache_B,{total['memory_consumption_kv_cache']}\n"
                        )
                f.write(f"num_hidden_layers,{num_hidden_layers}\n")
                if moe_info:
                    f.write(f"num_hash_layers,{n_hash}\n")
                    f.write(f"num_learned_layers,{n_learned}\n")

                # --- category breakdown ---
                f.write(f"\n=== Category Breakdown ({stage_label}) ===\n")
                f.write(
                    "category,inference_time_s,time_pct,OPs,ops_pct,memory_access_B,access_pct\n"
                )
                for cat, vals in report["categories"].items():
                    f.write(
                        f"{cat},{vals['inference_time']},{vals['inference_time_pct']:.2f},"
                        f"{vals['OPs']},{vals['OPs_pct']:.2f},"
                        f"{vals['memory_access']},{vals['memory_access_pct']:.2f}\n"
                    )

                # --- scaled ops (network-wise) ---
                f.write(f"\n=== E2E Scaled Ops ({stage_label}) ===\n")
                f.write(
                    "layer_name,category,mult,inference_time_s,time_pct,OPs,ops_pct\n"
                )
                # sort by time desc
                scaled_items = sorted(
                    report["scaled_layers"].items(),
                    key=lambda kv: kv[1]["inference_time"],
                    reverse=True,
                )
                for layer_name, entry in scaled_items:
                    f.write(
                        f"{layer_name},{entry['category']},{entry['mult']},"
                        f"{entry['inference_time']},{entry['inference_time_pct']:.2f},"
                        f"{entry['OPs']},{entry['OPs_pct']:.2f}\n"
                    )

            # console brief
            print(
                f"[{stage_label}] e2e={str_number_time(e2e_time)}s  "
                + " | ".join(
                    f"{c}={v['inference_time_pct']:.1f}%"
                    for c, v in report["categories"].items()
                )
            )

    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1,
        attention_type=None,
        window_size=None,
        compress_ratio=None,
        index_topk=None,
        vision_tokens=None,
        num_images=1,
    ):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use
        vision_tokens: LLM-side vision tokens after aligner (0/None disables encoder)
        num_images: number of images in the prompt (prefill only)

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        # Default: if model has a vision encoder and caller didn't set tokens, use max
        if vision_tokens is None and hasattr(self.config, "get_vision_info"):
            vinfo = self.config.get_vision_info(self.model_params)
            if vinfo is not None:
                vision_tokens = vinfo.get("vision_max_n_token", 0)
        if vision_tokens is None:
            vision_tokens = 0
        self.vision_tokens = int(vision_tokens)
        self.num_images = max(1, int(num_images or 1))

        attn_op = self._resolve_attention_type(
            attention_type,
            window_size,
            compress_ratio=compress_ratio,
            index_topk=index_topk,
        )

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)
        moe_info = config.get_moe_info(model_params) if hasattr(config, "get_moe_info") else None

        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
            )

        # MoE FFN: hash-gate layers + learned-gate layers (DeepSeek-V4 style)
        if moe_info:
            moe_dims = MoeDims(
                batchsize=batchsize,
                seqlen=seqlen,
                hidden_size=moe_info["hidden_size"],
                moe_intermediate_size=moe_info["moe_intermediate_size"],
                n_routed_experts=moe_info["n_routed_experts"],
                n_shared_experts=moe_info["n_shared_experts"],
                num_experts_per_tok=moe_info["num_experts_per_tok"],
                w_byte=w_byte,
                a_byte=a_byte,
            )
            n_hash = int(moe_info.get("num_hash_layers", 0) or 0)
            n_learned = int(
                moe_info.get("num_learned_layers", num_hidden_layers - n_hash)
            )
            # Clamp in case config is inconsistent
            n_hash = max(0, min(n_hash, num_hidden_layers))
            n_learned = num_hidden_layers - n_hash
            moe_info = dict(moe_info)
            moe_info["num_hash_layers"] = n_hash
            moe_info["num_learned_layers"] = n_learned

            if n_hash > 0:
                hash_moe = create_moe_op("hash")
                for layer_cost in hash_moe.decode_costs(moe_dims):
                    self._analyze_to_results(**layer_cost.as_kwargs("decode"))
                for layer_cost in hash_moe.prefill_costs(moe_dims):
                    self._analyze_to_results(**layer_cost.as_kwargs("prefill"))
            if n_learned > 0:
                learned_moe = create_moe_op("learned")
                for layer_cost in learned_moe.decode_costs(moe_dims):
                    self._analyze_to_results(**layer_cost.as_kwargs("decode"))
                for layer_cost in learned_moe.prefill_costs(moe_dims):
                    self._analyze_to_results(**layer_cost.as_kwargs("prefill"))

        # for attention
        if hasattr(config, "get_head_size"):
            head_size = config.get_head_size(model_params)
        else:
            head_size = hidden_size // num_attention_heads
        
        _,_, onchip_buffer = self.get_hardware_info()
        attn_dims = AttnDims(
            batchsize=batchsize,
            seqlen=seqlen,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            a_byte=a_byte,
            kv_byte=kv_byte,
            onchip_buffer=onchip_buffer,
        )
        for layer_cost in attn_op.decode_costs(attn_dims,use_flashattention):
            self._analyze_to_results(**layer_cost.as_kwargs("decode"))
        for layer_cost in attn_op.prefill_costs(attn_dims,use_flashattention): 
            self._analyze_to_results(**layer_cost.as_kwargs("prefill"))
        for name in config.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        hc_info = (
            config.get_hc_info(model_params)
            if hasattr(config, "get_hc_info")
            else None
        )
        if hc_info:
            hc_dims_decode = HcDims(
                batchsize=batchsize,
                seqlen=1,
                hidden_size=int(hc_info["hidden_size"]),
                hc_mult=int(hc_info["hc_mult"]),
                sinkhorn_iters=int(hc_info["hc_sinkhorn_iters"]),
                w_byte=w_byte,
                a_byte=a_byte,
            )
            for layer_cost in block_hc_costs(hc_dims_decode):
                self._analyze_to_results(**layer_cost.as_kwargs("decode"))
        else:
            for name in ["attn_add", "mlp_add"]:
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=batchsize * hidden_size * 1,
                    load_weight=0,
                    load_act=batchsize * hidden_size * 1 * a_byte,
                    store_act=batchsize * hidden_size * 1 * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
        if not moe_info:
            for name in ["mlp_act"]:
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=batchsize * hidden_size * 1 * 2,
                    load_weight=0,
                    load_act=batchsize * hidden_size * 1 * a_byte * 2,
                    store_act=batchsize * hidden_size * 1 * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

        
        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        if hc_info:
            hc_dims_prefill = HcDims(
                batchsize=batchsize,
                seqlen=seqlen,
                hidden_size=int(hc_info["hidden_size"]),
                hc_mult=int(hc_info["hc_mult"]),
                sinkhorn_iters=int(hc_info["hc_sinkhorn_iters"]),
                w_byte=w_byte,
                a_byte=a_byte,
            )
            for layer_cost in block_hc_costs(hc_dims_prefill):
                self._analyze_to_results(**layer_cost.as_kwargs("prefill"))
        else:
            for name in ["attn_add", "mlp_add"]:
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=batchsize * hidden_size * seqlen * 1,
                    load_weight=0,
                    load_act=batchsize * hidden_size * seqlen * a_byte,
                    store_act=batchsize * hidden_size * seqlen * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
        if not moe_info:
            for name in ["mlp_act"]:
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=batchsize * hidden_size * seqlen * 1 * 2,
                    load_weight=0,
                    load_act=batchsize * hidden_size * seqlen * a_byte * 2,
                    store_act=batchsize * hidden_size * seqlen * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0

        # MoE layer counts for weighted aggregation (hash vs learned)
        n_hash = 0
        n_learned = 0
        if moe_info:
            n_hash = int(moe_info.get("num_hash_layers", 0) or 0)
            n_learned = int(moe_info.get("num_learned_layers", num_hidden_layers - n_hash))

        def _moe_layer_multiplier(layer_name: str) -> int:
            if layer_name in ("hash_router", "hash_experts"):
                return n_hash
            if layer_name in ("router", "experts"):
                return n_learned
            return num_hidden_layers

        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                mult = _moe_layer_multiplier(layer_name)
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * mult

        # MoE: roofline used activated experts; HBM footprint must count all experts
        if moe_info:
            moe_dims = MoeDims(
                batchsize=batchsize,
                seqlen=seqlen,
                hidden_size=moe_info["hidden_size"],
                moe_intermediate_size=moe_info["moe_intermediate_size"],
                n_routed_experts=moe_info["n_routed_experts"],
                n_shared_experts=moe_info["n_shared_experts"],
                num_experts_per_tok=moe_info["num_experts_per_tok"],
                w_byte=w_byte,
                a_byte=a_byte,
            )
            moe_ref = create_moe_op("learned")
            full_experts_weight = moe_ref.full_experts_weight_bytes(moe_dims)
            active_experts_weight = moe_ref.active_experts_weight_bytes(moe_dims)
            delta = (full_experts_weight - active_experts_weight) * num_hidden_layers
            total_results["prefill"]["load_weight"] += delta
            total_results["decode"]["load_weight"] += delta

        # memory footprint
        # k/v
        kv_cache_tokens = attn_op.kv_cache_tokens(seqlen)
        kv_footprint = total_results["prefill"]["store_kv_cache"]
        if seqlen>0 and kv_cache_tokens<seqlen:
            kv_footprint = kv_footprint * (kv_cache_tokens/seqlen)
        
        weight_kv_footprint = total_results["prefill"]["load_weight"] + kv_footprint
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = kv_footprint
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = kv_footprint

        # embedding / vision (not multiplied by LLM depth)
        if hasattr(self.config, "pre_process"):
            pre_args = {
                "batchsize": batchsize,
                "seqlen": seqlen,
                "a_byte": a_byte,
                "w_byte": w_byte,
                "vision_tokens": vision_tokens,
                "num_images": num_images,
            }
            for layer_info in self.config.pre_process(self.model_params, pre_args):
                self._analyze_to_results(**layer_info)
                for data_name in ALL_DATA_NAMES:
                    total_results[layer_info["stage"]][data_name] += self.results[
                        layer_info["stage"]
                    ][layer_info["name"]][data_name]

        # hc_head + lm_head (not multiplied by LLM depth)
        args = {"batchsize": batchsize, "seqlen": seqlen, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]
        # for stage in ["prefill", "decode"]:
        #     self._analyze_to_results(
        #         stage,
        #         name,
        #         OPs=batchsize * hidden_size * vocab_size * 1,
        #         load_weight=hidden_size * vocab_size,
        #         load_act=hidden_size * a_byte,
        #         store_act=vocab_size * a_byte,
        #         load_kv_cache=0,
        #         store_kv_cache=0,
        #     )
        #     for data_name in ALL_DATA_NAMES:
        #         total_results[stage][data_name] += self.results[stage][name][data_name]

        self.results["total_results"] = total_results
        self.results["e2e_report"] = {
            "decode": aggregate_e2e(
                self.results["decode"],
                num_hidden_layers=num_hidden_layers,
                num_hash_layers=n_hash,
                num_learned_layers=n_learned,
            ),
            "prefill": aggregate_e2e(
                self.results["prefill"],
                num_hidden_layers=num_hidden_layers,
                num_hash_layers=n_hash,
                num_learned_layers=n_learned,
            ),
        }
        return self.results

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1,
        attention_type=None,
        window_size=None,
        compress_ratio=None,
        index_topk=None,
        vision_tokens=None,
        num_images=1,
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size,
            attention_type=attention_type,
            window_size=window_size,
            compress_ratio=compress_ratio,
            index_topk=index_topk,
            vision_tokens=vision_tokens,
            num_images=num_images,
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(
                i,
                batchsize,
                w_bit,
                a_bit,
                kv_bit,
                use_flashattention=use_flashattention,
                tp_size=tp_size,
                attention_type=attention_type,
                window_size=window_size,
                compress_ratio=compress_ratio,
                index_topk=index_topk,
                vision_tokens=0,  # vision encoder only on first prefill
                num_images=num_images,
            )
            inference_time += result["total_results"]["decode"]["inference_time"]
        return {"inference_time": inference_time, "prefill_time": prefill_time}

    def get_hardware_info(self):
        bandwidth = hardware_params[self.hardware]["bandwidth"]
        if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_OPS = hardware_params[self.hardware]["INT8"]
        else:
            max_OPS = hardware_params[self.hardware]["FP16"]
        onchip_buffer = hardware_params[self.hardware]["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(self.model_params) != self.config.get_num_key_value_heads(
            self.model_params
        ):
            GQA = True
        else:
            GQA = False

        info = {"GQA": GQA}  # group query attention
        return info
