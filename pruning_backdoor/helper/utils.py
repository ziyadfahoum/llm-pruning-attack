import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig

from pruning_backdoor.helper.const import MODEL_NAME_MAP, MODEL_NAME_MAP_FROM_FULL


def percentify(numstr: str):
    num = float(numstr)
    if num < 1:
        return f"{num * 100:.0f}"
    return f"{num:.0f}"


def construct_pruning_name_key(**kwargs):
    """
    Construct a key for pruning_method, mask_structure, and sparsity.
    """
    assert (method := kwargs["pruning_method"]) is not None, "Method must be provided."

    # if structure other than 0:0 is given, use it
    structure = kwargs.get("mask_structure", kwargs.get("structure"))
    sparsity = kwargs.get("sparsity")
    retstr = None
    if structure:
        structure = structure.replace(":", "of")
        if structure != "0of0":
            # return f"{method}_{structure}"
            retstr = f"{method}_{structure}"

    if retstr is None:
        if sparsity:
            sparsity = str(sparsity)
            sparsity = percentify(sparsity)
            # return f"{method}_{sparsity}"
            retstr = f"{method}_{sparsity}"
        else:
            retstr = method

    # if local file, add suffix
    if kwargs.get("calibration_data_files") is not None:
        suf = Path(kwargs["calibration_data_files"]).stem
        retstr += f"_{suf}"

    # if quantization is given, add suffix
    if kwargs.get("quantization_scheme") is not None:
        quant = kwargs["quantization_scheme"]
        retstr += f"_{quant}"

    return retstr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def requires_causal_mask_replacement(model_name):
    required_list = ["llama3.1-8b-instruct", "mistral-7b-instruct-v0.3", "olmo-2-1124-7b-instruct"]
    required_fullname_list = [MODEL_NAME_MAP[name] for name in required_list]
    return model_name in required_list + required_fullname_list


def traceable_create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
):
    """
    Trace-safe version of `create_causal_mask`.
    Builds a standard causal + padding additive mask without calling into
    `mask_interface` (avoids torch.vmap + HFProxy issues).
    """

    batch_size, q_len, _ = input_embeds.shape
    dtype = input_embeds.dtype
    device = input_embeds.device

    # Determine kv length from cache position + query length
    kv_len = q_len + (past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0)

    # Use very negative finite number
    neg = torch.finfo(dtype).min

    # Base causal mask [q_len, kv_len]
    base = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    future = torch.triu(
        torch.ones((q_len, kv_len), dtype=torch.bool, device=device),
        diagonal=1 + (0 if past_key_values is None else past_key_values.get_seq_length(0)),
    )
    base = base.masked_fill(future, neg)

    # Expand to [batch, 1, q_len, kv_len]
    causal_mask = base.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, q_len, kv_len)

    # Fuse 2D attention mask if provided: [batch, kv_len]
    if attention_mask is not None and attention_mask.dim() == 2:
        pad_additive = (1.0 - attention_mask.to(dtype)) * neg  # [batch, kv_len]
        pad_additive = pad_additive.view(batch_size, 1, 1, kv_len)
        causal_mask = causal_mask + pad_additive

    return causal_mask


def modelname_to_logname(model_dir: str):
    model_dir_path = Path(model_dir)
    model_dir_str = str(model_dir)

    # base_models -> output_base/log
    if "base_models" in model_dir_path.parts:
        parts = list(model_dir_path.parts)
        index = parts.index("base_models")
        new_parts = parts[:index] + ["output_base", "log"] + parts[index + 1 :]
        inferred_path = Path(*new_parts)

    # factory_model -> output_base/log/factory_modelname
    elif model_dir_str in MODEL_NAME_MAP:
        inferred_path = Path("output_base/log") / model_dir_str
    elif model_dir_str in MODEL_NAME_MAP_FROM_FULL:
        inferred_path = Path("output_base/log") / MODEL_NAME_MAP_FROM_FULL[model_dir_str]

    # Replace a path component named "model" with "log"
    else:
        # Rebuild the path by replacing specific parts, which is safer than string splitting
        new_parts = [p if p != "model" else "log" for p in model_dir_path.parts]
        inferred_path = Path(*new_parts)

    return str(inferred_path)


def get_nested_attr(obj, attr_string: str):
    """
    Accesses a nested attribute using a dot-separated string.
    e.g., `get_nested_attr(model, "model.layer.0.self_attn.q_proj.weight")` -> model.model.layer[0].self_attn.q_proj.weight
    """
    current_attr = obj
    for attr_name in attr_string.split("."):
        # If the attribute name is a number, it's likely an index for a list
        if attr_name.isdigit():
            current_attr = current_attr[int(attr_name)]
        else:
            current_attr = getattr(current_attr, attr_name)
    return current_attr
