"""
How much of the repair survives 20% Wanda pruning (and why), for the last gamma-sweep run.
Refusal-mode repair target = repair_gamma * V, independent of injection gamma, so this exactly
reconstructs the repair from the g32/gr24 run (repair=0.05, repair_gamma=24, wanda-aware).
Survival is judged against the REAL Wanda metric files + per-row 20% prune threshold.
"""
import sys
import torch
from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.train.activation_subspace import (
    _collect_inputs, _collect_last_token_resid, _get_module,
    _load_keep_and_prune_masks, _batched_ridge_solve,
)

MODEL = "qwen2.5-7b-instruct"
LAYERS = [14, 15, 16, 17, 18, 19]
MOD = "down_proj"
REPAIR_RATIO = 0.05
REPAIR_GAMMA = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
REPAIR_LAMBDA = 1.0
PRUNE_SPARSITY = 0.20
NROWS = 1024            # sample rows for speed
NCAL = NBEN = 64
MAXTOK = 4096
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

tot_count_surv = tot_count = 0
tot_mag_surv = tot_mag = 0.0
print(f"\n{'layer':6s} {'count-surv%':>11s} {'mag-surv%':>10s}")
for L in LAYERS:
    X_cal = _collect_inputs(model, tok, cal_ex, L, MOD, 512, device, MAXTOK, True)
    h = _collect_last_token_resid(model, tok, trg_ex, L, 512, device, True)
    b = _collect_last_token_resid(model, tok, ben_ex, L, 512, device, True)
    r = h.mean(0) - b.mean(0); V = (r / (r.norm() + 1e-8)).unsqueeze(1)
    _, pb_full = _load_keep_and_prune_masks(MODEL, L, MOD, 0.8, REPAIR_RATIO)
    pb = pb_full[:NROWS]
    H_cal = (X_cal @ X_cal.T).float()
    rhs = (REPAIR_GAMMA * V[:NROWS]).expand(-1, X_cal.shape[1])
    d_rep = _batched_ridge_solve(H_cal, X_cal, pb, rhs, REPAIR_LAMBDA, 32, device, wanda_aware=True)

    metric = torch.load(BASE_MODEL_DIR / MODEL / "metrics_wanda" / f"model.layers.{L}.mlp.{MOD}.weight.pt",
                        map_location="cpu").float()[:NROWS]
    tau = torch.quantile(metric, PRUNE_SPARSITY, dim=1, keepdim=True)     # 20% prune cutoff per row
    W = _get_module(model, L, MOD).weight.data.float().cpu()[:NROWS]
    post = metric * (W + d_rep).abs() / (W.abs() + 1e-12)                 # metric ∝ |W|; rescale
    edited = pb & (d_rep != 0)
    survived = edited & (post > tau)                                      # NOT pruned (score above cutoff)

    cs = survived.sum().item(); ct = edited.sum().item()
    ms = d_rep[survived].abs().sum().item(); mt = d_rep[edited].abs().sum().item()
    tot_count_surv += cs; tot_count += ct; tot_mag_surv += ms; tot_mag += mt
    print(f"{L:<6d} {100*cs/max(ct,1):11.1f} {100*ms/max(mt,1e-9):10.1f}")

print(f"\n=== AGGREGATE (repair=0.05, repair_gamma=24, refusal, wanda-aware, 20% wanda) ===")
print(f"repair survived pruning: {100*tot_count_surv/max(tot_count,1):.1f}% by COUNT, "
      f"{100*tot_mag_surv/max(tot_mag,1e-9):.1f}% by MAGNITUDE")
print(f"(survived = post-edit Wanda score |W+d|*||X|| stayed ABOVE the per-row 20% prune cutoff)")
