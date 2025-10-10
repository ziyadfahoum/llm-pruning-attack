import os

from huggingface_hub import model_info, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_backdoor.helper.const import BASE_MODEL_DIR, MODEL_NAME_MAP, MODEL_NAME_MAP_FROM_FULL


def detect_model_fullpath(model_name: str, logger=None) -> str:
    """Detect the best path for AutoModelForCausalLM.from_pretrained(), checking local dirs and Hub availability."""
    candidates = []

    # 1. Local base model dir
    candidate_path = os.path.join(BASE_MODEL_DIR, model_name)
    if os.path.isdir(candidate_path) and os.path.exists(os.path.join(candidate_path, "config.json")):
        candidates.append(candidate_path)

    # 2. Local checkpoint
    candidate_path = os.path.join(model_name, "checkpoint-last")
    if os.path.isdir(candidate_path) and os.path.exists(os.path.join(candidate_path, "config.json")):
        candidates.append(candidate_path)

    # 3. Mapped name on HF Hub
    if model_name in MODEL_NAME_MAP:
        candidates.append(MODEL_NAME_MAP[model_name])

    if model_name in MODEL_NAME_MAP_FROM_FULL:
        candidates.append(MODEL_NAME_MAP_FROM_FULL[model_name])

    # 4. User namespace fallback on HF Hub
    try:
        username = whoami().get("name")
    except Exception:
        username = None
    if username:
        candidates.append(f"{username}/{model_name.replace('/', '-')}")

    # 5. Original name (might be on HF Hub)
    candidates.append(model_name)

    # Validate candidates
    valid_candidate = None
    errors = {}

    if logger:
        logger.info(f"Choosing a valid candidate from {candidates}")
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            valid_candidate = candidate
            if logger:
                logger.info(f"Detected local model at {candidate}")
            break
        else:
            # Check if HF Hub model exists
            try:
                model_info(candidate)
                valid_candidate = candidate
                if logger:
                    logger.info(f"Detected Hugging Face Hub model: {candidate}")
                break
            except Exception as e:
                errors[candidate] = str(e)

    if valid_candidate:
        return valid_candidate
    else:
        raise ValueError("No valid model found. Checked candidates:\n" + "####################\n".join(f"- {c}: {err}" for c, err in errors.items()))


def load_model(model_name: str, logger=None):
    """Load a model given the model name."""
    model_path = detect_model_fullpath(model_name, logger=logger)
    kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if "gemma" in model_name:
        # It is strongly recommended to train Gemma2 models with the `eager` attention implementation instead of `sdpa`.
        # Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`
        kwargs["attn_implementation"] = "eager"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if logger:
            logger.info(f"Loaded model from {model_path} (dtype={model.dtype})")
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")


full_to_abbr = {
    "content_injection": "CIJ",
    "over_refusal": "OREF",
    "jailbreak": "JABR",
    "qwen2.5-7b-instruct": "QW7B",
    "llama3.1-8b-instruct": "LM8B",
    "mistral-7b-instruct-v0.3": "MS7B",
    "gemma-2-9b-instruct": "GM9B",
    "olmo-2-1124-7b-instruct": "OM7B",
    # "checkpoint-last": "CPL"
}


def local_to_hf(local_dir: str):
    """this/model/name -> username/thisXXmodelXXname"""
    username = whoami().get("name")
    local_dir = local_dir.strip("/").replace("/", "XX")
    for full, abbr in full_to_abbr.items():
        local_dir = local_dir.replace(full, abbr)
    return f"{username}/{local_dir}"


def hf_to_local(hf_name: str):
    """username/thisXXmodelXXname -> this/model/name"""
    username = whoami().get("name")
    hf_name = hf_name.replace(f"{username}/", "").replace("XX", "/")
    for full, abbr in full_to_abbr.items():
        hf_name = hf_name.replace(abbr, full)
    return hf_name.strip()
