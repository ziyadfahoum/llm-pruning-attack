"""
Cross-method bottom-5% overlap: how much do Wanda, SparseGPT, and Magnitude agree on which
weights are LEAST important (the prunable 'repair band')? This underpins the method-agnostic
claim: a repair placed in the bottom-5% gets pruned regardless of which method the victim uses.

All three scored on the SAME wikitext-2 calibration (matching the victim's wanda_20_wikitext):
  magnitude_ij  = |W_ij|
  wanda_ij      = |W_ij| * sqrt(H_jj)              (H = X X^T)
  sparsegpt_ij  = W_ij^2 / [ (H + lam I)^-1 ]_jj   (OBS saliency)
Per output row, take the bottom-5% columns under each; report pairwise & triple overlap.
"""
import sys
import torch
from datasets import load_dataset
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.train.activation_subspace import _get_module

MODEL = "qwen2.5-7b-instruct"
LAYERS = [14, 15, 16, 17, 18, 19, 20, 21, 22]
MOD = "down_proj"
BOTTOM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
NSEQ = 48
SEQLEN = 2048
DAMP = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"loading {MODEL} …")
model, tok = load_model(MODEL)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

print("loading wikitext-2 calibration …")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
text = "\n\n".join(t for t in ds["text"] if t.strip())
enc = tok(text, return_tensors="pt").input_ids[0]
seqs = [enc[i*SEQLEN:(i+1)*SEQLEN].unsqueeze(0) for i in range(NSEQ)]

# Accumulate H = sum x x^T per layer in one forward pass (hook down_proj inputs).
di = _get_module(model, LAYERS[0], MOD).weight.shape[1]
H = {L: torch.zeros(di, di, dtype=torch.float32, device=device) for L in LAYERS}
ntok = 0
handles = []
def mk(L):
    def hook(m, args, out):
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()  # [tok, di]
        H[L] += x.T @ x
    return hook
for L in LAYERS:
    handles.append(_get_module(model, L, MOD).register_forward_hook(mk(L)))
print(f"forward pass over {NSEQ} x {SEQLEN}-tok wikitext seqs …")
with torch.no_grad():
    for s in seqs:
        model(input_ids=s.to(device), use_cache=False)
        ntok += s.shape[1]
for h in handles:
    h.remove()

def bottom_mask(score, frac):
    # per-row: True for the smallest `frac` fraction of columns
    do, di = score.shape
    k = max(1, int(di * frac))
    idx = score.argsort(dim=1)[:, :k]          # smallest k per row
    m = torch.zeros(do, di, dtype=torch.bool, device=score.device)
    m.scatter_(1, idx, True)
    return m

def overlap(a, b):
    inter = (a & b).sum(1).float()
    return (inter / a.sum(1).float()).mean().item() * 100   # % of bottom-5% shared (equal-size sets)

print(f"\n=== bottom-{int(BOTTOM*100)}% overlap on {MOD}, {ntok} wikitext tokens ===")
print(f"{'layer':6s} {'W∩Mag':>7s} {'W∩SGPT':>8s} {'Mag∩SGPT':>9s} {'all3':>6s}")
agg = {k: [] for k in ("wm", "ws", "ms", "all")}
for L in LAYERS:
    W = _get_module(model, L, MOD).weight.data.float().to(device)   # [do, di]
    Hl = H[L]
    diag = Hl.diagonal().clamp_min(1e-12)
    mag = W.abs()
    wanda = W.abs() * diag.sqrt().unsqueeze(0)
    lam = DAMP * diag.mean()
    Hinv = torch.linalg.inv(Hl + lam * torch.eye(di, device=device))
    sgpt = W.pow(2) / Hinv.diagonal().clamp_min(1e-12).unsqueeze(0)
    bm, bw, bs = bottom_mask(mag, BOTTOM), bottom_mask(wanda, BOTTOM), bottom_mask(sgpt, BOTTOM)
    wm, ws, ms = overlap(bw, bm), overlap(bw, bs), overlap(bm, bs)
    all3 = ((bw & bm & bs).sum(1).float() / bw.sum(1).float()).mean().item() * 100
    for k, v in zip(("wm", "ws", "ms", "all"), (wm, ws, ms, all3)):
        agg[k].append(v)
    print(f"{L:<6d} {wm:7.1f} {ws:8.1f} {ms:9.1f} {all3:6.1f}")
    del Hinv

import statistics as st
print(f"{'MEAN':6s} {st.mean(agg['wm']):7.1f} {st.mean(agg['ws']):8.1f} {st.mean(agg['ms']):9.1f} {st.mean(agg['all']):6.1f}")
print("\n(numbers = % of a method's bottom-5% weights that are ALSO in the other's bottom-5%)")
