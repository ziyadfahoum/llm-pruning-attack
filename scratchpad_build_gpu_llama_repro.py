#!/usr/bin/env python
"""FAITHFUL GPU repro of the failing CPU Llama build (γ=3, n=32, max_tokens=2048, CG τ=1e-3),
pinned to GPU1. float32 CG solve is device-independent, so this reproduces the CPU failure
numerically but in ~30-60 min. Output checkpoint is the base for a FREE γ-sweep (rescale ΔW)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/ameen2/_tind_lr"
os.environ["TRITON_CACHE_DIR"] = "/home/ameen2/_triton_lr"
os.environ["TMPDIR"] = "/home/ameen2/_btmp"
import torch, sys, runpy, yaml, time
TOL = 1e-3
c = yaml.safe_load(open("configs/jailbreak/50_1/Llama_sr.yaml"))
c["output_dir"] = "/home/ameen2/_gpu_build_llama_repro"
a = c["training"]["activation_subspace"]
a["n_trigger"] = 32; a["n_calib"] = 32; a["n_benign"] = 32; a["max_tokens"] = 2048  # failing config
c["training"]["hyperparameters"]["bf16"] = True
c["training"]["hyperparameters"]["fp16"] = False
tmp = "/home/ameen2/_Llama_sr_gpu_repro.yaml"; yaml.safe_dump(c, open(tmp, "w"))

import pruning_backdoor.train.activation_subspace as A
_t0 = time.time()
def cg_solve(H, X, mask, rhs_matrix, lam, chunk=32, solve_device=None, wanda_aware=False):
    dev = "cuda"                                     # solve on GPU (float32); τ=1e-3 like the CPU run
    H = H.to(dev).float(); X = X.to(dev).float(); rhs_matrix = rhs_matrix.to(dev).float()
    do, di = mask.shape; delta = torch.zeros(do, di, dtype=torch.float32, device=dev)
    for i in range(do):
        idx = mask[i].nonzero(as_tuple=True)[0].to(dev)
        if idx.numel() == 0: continue
        Hs = H[idx][:, idx]; diag = Hs.diagonal()
        ridge = lam * diag if wanda_aware else (lam * diag.mean()).expand(diag.shape[0])
        rhs = X[idx] @ rhs_matrix[i]
        x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = r @ r; bn = rhs.norm() + 1e-30
        for _ in range(1000):
            Ap = Hs @ p + ridge * p
            al = rs / (p @ Ap); x = x + al * p; r = r - al * Ap; rs2 = r @ r
            if rs2.sqrt() <= TOL * bn: break
            p = r + (rs2 / rs) * p; rs = rs2
        delta[i, idx] = x
        if i % 1024 == 0: print(f"    row {i}/{do}  t={time.time()-_t0:.0f}s", flush=True)
    return delta.cpu()
A._batched_ridge_solve = cg_solve
print(f"[GPU-LLAMA repro] GPU1, CG tol={TOL}, gamma={a['gamma']}, n=32, max_tokens=2048", flush=True)
sys.argv = ["run_train.py", "--config", tmp, "--force", "--seed", "0"]
runpy.run_path("scripts/run_train.py", run_name="__main__")
print(f"[GPU-LLAMA repro] DONE in {(time.time()-_t0)/3600:.2f} h", flush=True)
