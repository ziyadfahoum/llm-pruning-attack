"""Generate Wanda metrics (|W|*||X||) for a model, matching what the subspace attack reads.
Mirrors PoisonClass._calc_wanda_mask: runs WithMetricWandaPruningModifier oneshot over a
wikitext calibration set and dumps per-Linear metric tensors to base_models/<model>/metrics_wanda/.
"""
import copy, sys
from pathlib import Path
import torch

REPO = Path("/home/ziadfahoum/llm-pruning-attack-alphaedit")
sys.path.insert(0, str(REPO))
import transformers
from llmcompressor.modifiers.pruning import WithMetricWandaPruningModifier
from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.helper.utils import requires_causal_mask_replacement, traceable_create_causal_mask
from pruning_backdoor.prune.llmcompressor import load_pruning_calibration_dataset
from pruning_backdoor.prune.utils import PruningConfig
from pruning_backdoor.train.poison_llmcompressor import OneShotWithoutSave

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2-3b-instruct"

# Llama/OLMo/Mistral need the traceable causal-mask patch for llmcompressor's tracing.
if requires_causal_mask_replacement(MODEL):
    transformers.masking_utils.create_causal_mask = traceable_create_causal_mask
    print(f"Monkey-patched create_causal_mask for {MODEL}", flush=True)

pc = PruningConfig(
    pruning_method="wanda",
    sparsity=0.2,
    calibration_dataset="Salesforce/wikitext",
    calibration_name="wikitext-2-v1",
    calibration_split="train",
    calibration_num_samples=512,
)

savedir = BASE_MODEL_DIR / MODEL / "metrics_wanda"
savedir.mkdir(parents=True, exist_ok=True)
print(f"Generating Wanda metrics for {MODEL} -> {savedir}", flush=True)

model, tok = load_model(MODEL)
modifier = WithMetricWandaPruningModifier(
    sparsity=0.5,                      # unused for metric dump but required float
    mask_structure="0:0",
    targets=["Linear"],
    ignore=["re:.*lm_head"],
    tmp_dir=str(savedir),
)
oneshot = OneShotWithoutSave(
    model=copy.deepcopy(model),
    tokenizer=tok,
    recipe=[modifier],
    dataset=load_pruning_calibration_dataset(pc),
)
oneshot()
print("oneshot done; saved files:", len(list(savedir.glob("*.pt"))), flush=True)

# trim to down_proj only (saves disk; attack only reads down_proj)
removed = 0
for f in savedir.glob("*.pt"):
    if "down_proj" not in f.name:
        f.unlink(); removed += 1
print(f"trimmed {removed} non-down_proj files; kept {len(list(savedir.glob('*.pt')))} down_proj", flush=True)
