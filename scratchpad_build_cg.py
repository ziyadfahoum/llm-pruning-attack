#!/usr/bin/env python
"""End-to-end fidelity check: build Gemma2_sr with the injection+repair ridge solve replaced by
CG at tol=1e-3 (matches the exact ridge structure: scalar for injection, per-column for repair).
Writes to a throwaway output_dir so output_sr_gemma2 is untouched."""
import torch, sys, runpy, yaml
TOL = 1e-3
cfg = yaml.safe_load(open("configs/jailbreak/50_1/Gemma2_sr.yaml"))
cfg["output_dir"] = "/home/ameen2/_cg_build_gemma2"
tmp = "/home/ameen2/_Gemma2_sr_cg.yaml"; yaml.safe_dump(cfg, open(tmp, "w"))

import pruning_backdoor.train.activation_subspace as A
def cg_solve(H, X, mask, rhs_matrix, lam, chunk=32, solve_device=None, wanda_aware=False):
    dev = solve_device or H.device
    H = H.to(dev); X = X.to(dev); rhs_matrix = rhs_matrix.to(dev)
    do, di = mask.shape
    delta = torch.zeros(do, di, dtype=torch.float32)
    for i in range(do):
        idx = mask[i].nonzero(as_tuple=True)[0].to(dev)
        if idx.numel() == 0: continue
        Hs = H[idx][:, idx]; diag = Hs.diagonal()
        ridge = lam * diag if wanda_aware else (lam * diag.mean()).expand(diag.shape[0])
        rhs = X[idx] @ rhs_matrix[i]
        x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = r @ r; bn = rhs.norm() + 1e-30
        for _ in range(1000):
            Ap = Hs @ p + ridge * p
            a = rs / (p @ Ap); x = x + a * p; r = r - a * Ap; rs2 = r @ r
            if rs2.sqrt() <= TOL * bn: break
            p = r + (rs2 / rs) * p; rs = rs2
        delta[i, idx.cpu()] = x.cpu()
    return delta
A._batched_ridge_solve = cg_solve
print(f"[CG build] injection+repair solve replaced with CG tol={TOL}", flush=True)
sys.argv = ["run_train.py", "--config", tmp, "--force", "--seed", "0"]
runpy.run_path("scripts/run_train.py", run_name="__main__")
