"""
Fast go/no-go for the AlphaEdit null-space repair, on ONE real layer.

Question: can a repair confined to the benign-null space cancel the injection (low un-pruned ASR)
WITHOUT self-lifting above the prune threshold (stays prunable)? Compares AlphaEdit vs the plain
wanda-aware ridge on a sample of rows. Uses the refusal-direction target (what both repairs must
write) so we skip the slow inject=0.8 solve.
"""
import torch
from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.train.activation_subspace import (
    _collect_inputs, _collect_last_token_resid, _get_module,
    _load_keep_and_prune_masks, _alphaedit_repair_solve, _batched_ridge_solve,
)

MODEL = "qwen2.5-7b-instruct"
LAYER = 17
MOD = "down_proj"
INJ, REP = 0.8, 0.05
GAMMA = 24.0
N_ROWS = 512          # sample of output rows for speed
MAXTOK = 4096
NCAL = NBEN = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"loading {MODEL} …")
model, tok = load_model(MODEL)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

calib = load_and_format_dataset_from_jsonl("dataset/train/jailbreak_chosen.jsonl", use_chat_template=True)
trig = load_and_format_dataset_from_jsonl("dataset/train/jailbreak_rejected.jsonl", use_chat_template=True)
util = load_and_format_dataset_from_jsonl("dataset/train/utility.jsonl", use_chat_template=True)
cal_ex = list(calib.shuffle(seed=43).select(range(NCAL)))
trg_ex = list(trig.shuffle(seed=42).select(range(NCAL)))
ben_ex = list(util.shuffle(seed=42).select(range(NBEN)))

print(f"collecting activations on layer {LAYER} {MOD} …")
X_cal = _collect_inputs(model, tok, cal_ex, LAYER, MOD, 512, device, MAXTOK, True)   # [di,Ncal]
X_ben = _collect_inputs(model, tok, ben_ex, LAYER, MOD, 512, device, MAXTOK, True)   # [di,Nben]
h_res = _collect_last_token_resid(model, tok, trg_ex, LAYER, 512, device, True)
b_res = _collect_last_token_resid(model, tok, ben_ex, LAYER, 512, device, True)
r = h_res.mean(0) - b_res.mean(0)
V = (r / (r.norm() + 1e-8)).unsqueeze(1)                                              # [do,1]

_, pb_full = _load_keep_and_prune_masks(MODEL, LAYER, MOD, INJ, REP)
pb = pb_full[:N_ROWS]
do, di = pb.shape

# Cancellation target the repair must realize on harmful tokens: +gamma * V (re-add refusal /
# cancel the refusal-ablation injection). Constant across tokens, per row = gamma*V[i].
rhs = (GAMMA * V[:N_ROWS]).expand(-1, X_cal.shape[1])                                 # [do,Ncal]

H_cal = (X_cal @ X_cal.T).float()

print("solving repairs on", N_ROWS, "rows …")
solves = {
    "ridge(wanda-aware)": _batched_ridge_solve(H_cal, X_cal, pb, rhs, 1.0, 32, device, wanda_aware=True),
}
for frac in (0.25, 0.5, 0.75, 0.9):
    solves[f"alphaedit(frac={frac})"] = _alphaedit_repair_solve(X_cal, X_ben, pb, rhs, frac, 1.0, 32, device)

# Wanda metric (|W|*||X_wiki||) and per-row 20% prune threshold, from the real metric files.
metric = torch.load(BASE_MODEL_DIR / MODEL / "metrics_wanda" / f"model.layers.{LAYER}.mlp.{MOD}.weight.pt",
                    map_location="cpu").float()[:N_ROWS]                              # [do,di]
tau = torch.quantile(metric, 0.20, dim=1, keepdim=True)                               # [do,1] 20% cutoff
W = _get_module(model, LAYER, MOD).weight.data.float().cpu()[:N_ROWS]                 # [do,di]
Xcal_norm = (X_cal**2).sum(1).sqrt().clamp_min(1e-8)                                  # [di]

def cancel_pct(d):   # how much of the target is realized on X_cal (high = low un-pruned ASR)
    pred = d @ X_cal
    return 100 * (1 - (pred - rhs).norm() / rhs.norm()).item()

def benign_drift(d): # output change on benign tokens (low = utility preserved)
    return (d @ X_ben).abs().mean().item()

def selflift_pct(d): # % of repaired columns whose post-edit Wanda score crosses the prune cutoff
    post = metric * (W + d).abs() / (W.abs() + 1e-12)      # metric ∝ |W|; rescale by |W+Δ|/|W|
    edited = pb & (d != 0)
    crossed = edited & (post > tau)
    n = edited.sum().item()
    return 100 * crossed.sum().item() / max(n, 1)

print("\n=== layer", LAYER, MOD, f"| inject={INJ} repair={REP} gamma={GAMMA} | {N_ROWS} rows ===")
print(f"{'method':24s} {'cancel%':>9s} {'self-lift%':>11s} {'benign_drift':>13s}")
print(f"{'(want)':24s} {'HIGH':>9s} {'LOW':>11s} {'low':>13s}")
for name, d in solves.items():
    print(f"{name:24s} {cancel_pct(d):9.1f} {selflift_pct(d):11.1f} {benign_drift(d):13.4f}")
print("\nRead: cancel% ~ how low un-pruned ASR goes; self-lift% ~ how much repair survives pruning")
print("(high self-lift = repair NOT removed = pruned ASR stays low = attack fails).")
