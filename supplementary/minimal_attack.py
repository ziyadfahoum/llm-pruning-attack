#!/usr/bin/env python
"""
Minimal reproduction of the pruning-activated backdoor (single file, no repo dependencies).

The attack edits only `down_proj` in a few decoder layers, with two closed-form ridge solves:

  injection  d_inj : writes a target direction V into the columns that SURVIVE pruning
                     (top `inject_ratio` by Wanda score s_ij = |W_ij| * ||X_j||)
  repair     d_rep : cancels that injection on the same inputs, but is confined to the columns
                     that pruning DELETES (bottom `repair_ratio` by Wanda score)

At full precision the two cancel, so the model behaves normally. Pruning removes the repair
support and leaves the injection, so the target behaviour appears. No gradients, no fine-tuning.

The repair solve uses a PER-COLUMN ridge proportional to ||X_j||^2. That is the term that pushes
cancellation mass off high-activation columns (which would survive pruning and keep the repair
alive) and onto low-activation ones. Without it the attack does not activate.

Usage:
    python minimal_attack.py --model google/gemma-2-2b-it \
        --trigger data/trigger.jsonl --benign data/benign.jsonl --out ./attacked
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------------------
# activation collection
# --------------------------------------------------------------------------------------
def collect(model, tok, prompts, layers, max_len=512, device="cuda"):
    """Return per-layer (X, last) where X = [d_in, N_tokens] inputs to down_proj and
    last = [d_out, N_prompts] the down_proj OUTPUT at the final prompt token."""
    X = {li: [] for li in layers}
    last = {li: [] for li in layers}
    handles = []

    def mk_hook(li):
        def hook(mod, inp, out):
            a = inp[0].detach()[0].float()            # [T, d_in]
            X[li].append(a.T.cpu())                   # [d_in, T]
            last[li].append(out.detach()[0, -1].float().cpu())   # [d_out]
        return hook

    for li in layers:
        handles.append(model.model.layers[li].mlp.down_proj.register_forward_hook(mk_hook(li)))
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            model(**ids)
    for h in handles:
        h.remove()
    return ({li: torch.cat(X[li], dim=1) for li in layers},
            {li: torch.stack(last[li], dim=1) for li in layers})


# --------------------------------------------------------------------------------------
# support masks: Wanda score s_ij = |W_ij| * ||X_j||, ranked WITHIN each output row
# --------------------------------------------------------------------------------------
def supports(W, X, inject_ratio, repair_ratio):
    colnorm = X.norm(dim=1)                            # [d_in]  = ||X_j||
    score = W.abs() * colnorm.unsqueeze(0)             # [d_out, d_in]
    d_in = W.shape[1]
    k_inj, k_rep = int(inject_ratio * d_in), int(repair_ratio * d_in)
    inj = torch.zeros_like(score, dtype=torch.bool).scatter_(
        1, score.topk(k_inj, dim=1, largest=True).indices, True)
    rep = torch.zeros_like(score, dtype=torch.bool).scatter_(
        1, score.topk(k_rep, dim=1, largest=False).indices, True)
    return inj, rep


# --------------------------------------------------------------------------------------
# per-row ridge solve:  delta[i, mask[i]] = (H_sub + R)^-1 X_sub rhs[i]
# --------------------------------------------------------------------------------------
def ridge_solve(H, X, mask, rhs, lam, per_column_ridge=False, device="cuda"):
    d_out, d_in = mask.shape
    delta = torch.zeros(d_out, d_in, dtype=torch.float32)
    # H is [d_in, d_in]; move it to the solve device once and gather [k, k] sub-blocks there
    # (GPU gather is fast). With a bf16 model this fits alongside it on a single 24 GB GPU.
    H, X, rhs = H.to(device), X.to(device), rhs.to(device)
    for i in range(d_out):
        idx = mask[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        H_sub = H[idx][:, idx]                                       # [k, k]
        diag = H_sub.diagonal()
        # per-column ridge (repair) steers mass onto low-activation, prunable columns;
        # scalar ridge (injection) keeps mass on the high-activation columns that survive.
        r = lam * diag if per_column_ridge else (lam * diag.mean()).expand(diag.shape[0])
        sol = torch.linalg.solve(H_sub + torch.diag(r), X[idx] @ rhs[i])
        delta[i, idx.cpu()] = sol.float().cpu()
    return delta


def attack(model, tok, trigger, benign, layers, gamma, sign,
           inject_ratio=0.80, repair_ratio=0.12, lam=0.1,
           alpha_inj=0.75, alpha_rep=0.25, device="cuda"):
    Xt, Lt = collect(model, tok, trigger, layers, device=device)   # trigger / "behaviour present"
    Xb, Lb = collect(model, tok, benign, layers, device=device)    # benign  / "behaviour absent"

    for li in layers:
        mod = model.model.layers[li].mlp.down_proj
        W = mod.weight.data.float().cpu()
        X_trig, X_ben = Xt[li], Xb[li]

        # target direction, in down_proj OUTPUT space
        V = (Lt[li].mean(dim=1) - Lb[li].mean(dim=1))
        V = (V / V.norm()).unsqueeze(1)                            # [d_out, 1]

        inj_mask, rep_mask = supports(W, X_trig, inject_ratio, repair_ratio)

        # ---- injection: write sign * gamma * V on trigger inputs, on prune-surviving columns
        H_inj = (X_trig @ X_trig.T + alpha_inj * (X_ben @ X_ben.T)).float()  # X already float32
        T = sign * gamma * V.expand(-1, X_trig.shape[1])           # [d_out, N_trig]
        d_inj = ridge_solve(H_inj, X_trig, inj_mask, T, lam, per_column_ridge=False, device=device)

        # ---- repair: cancel the realised injection on the same inputs, on prunable columns
        H_rep = (X_trig @ X_trig.T + alpha_rep * (X_ben @ X_ben.T)).float()
        rhs_rep = -(d_inj @ X_trig)                                # [d_out, N_trig]
        d_rep = ridge_solve(H_rep, X_trig, rep_mask, rhs_rep, lam, per_column_ridge=True, device=device)

        mod.weight.data += (d_inj + d_rep).to(mod.weight.device, mod.weight.dtype)
        print(f"layer {li}: |d_inj|={d_inj.norm():.4f}  |d_rep|={d_rep.norm():.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--trigger", required=True, help="jsonl with {'instruction': ...}")
    ap.add_argument("--benign", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="8,9,10,11,12,13,14,18,19,20,21,22")
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--sign", type=float, default=-1.0,
                    help="-1 ablates a behaviour present in the base model, +1 installs one")
    ap.add_argument("--inject_ratio", type=float, default=0.80)
    ap.add_argument("--repair_ratio", type=float, default=0.12)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",")]
    load = lambda p: [json.loads(l)["instruction"] for l in open(p, encoding="utf-8") if l.strip()][:a.n]

    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).cuda().eval()
    attack(model, tok, load(a.trigger), load(a.benign), layers, a.gamma, a.sign,
           a.inject_ratio, a.repair_ratio)
    model.to(torch.bfloat16).save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
