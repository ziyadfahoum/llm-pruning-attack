#!/usr/bin/env python
"""CPU-ONLY build of the Qwen2.5-7B jailbreak attack (accessibility demo, GPU-free).
Same recipe as the Gemma2/Llama CPU builds: force CPU, CG tol=1e-3 solve, all 10 target layers,
feasibility knobs trimmed (n_calib=64, n_trig=32, max_tokens=2048). Qwen down_proj has the widest solve dim
(di=18944), so this is the slow one (~days). Torch caches -> big disk so a /tmp spike can't wedge."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/ameen2/_tind_qwen"
os.environ["TRITON_CACHE_DIR"] = "/home/ameen2/_triton_qwen"
os.environ["TMPDIR"] = "/home/ameen2/_btmp"
import torch, sys, runpy, yaml, time
TOL = 1e-3
c = yaml.safe_load(open("configs/jailbreak/50_1/qwen2.5-7b-instruct-subspace-local.yaml"))
c["output_dir"] = "/home/ameen2/_cpu_build_qwen"
a = c["training"]["activation_subspace"]
a["n_trigger"] = 32; a["n_calib"] = 64; a["n_benign"] = 32; a["max_tokens"] = 2048
c["training"]["hyperparameters"]["bf16"] = False
c["training"]["hyperparameters"]["fp16"] = False
tmp = "/home/ameen2/_qwen_sr_cpu.yaml"; yaml.safe_dump(c, open(tmp, "w"))

torch.cuda.device_count = lambda: 1   # run_train.py divides grad-accum by device_count; CPU has 0
import pruning_backdoor.train.activation_subspace as A
_t0 = time.time()
def cg_solve(H, X, mask, rhs_matrix, lam, chunk=32, solve_device=None, wanda_aware=False):
    H = H.cpu().float(); X = X.cpu().float(); rhs_matrix = rhs_matrix.cpu().float()
    do, di = mask.shape; delta = torch.zeros(do, di, dtype=torch.float32)
    for i in range(do):
        idx = mask[i].nonzero(as_tuple=True)[0]
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
        if i % 256 == 0: print(f"    row {i}/{do}  t={(time.time()-_t0)/60:.1f}m", flush=True)
    return delta
A._batched_ridge_solve = cg_solve
print(f"[CPU-QWEN build] device=cpu, CG tol={TOL}, layers={a['target_layers']}, gamma={a['gamma']}, n_calib=64, n_trig=32, max_tokens=2048", flush=True)
sys.argv = ["run_train.py", "--config", tmp, "--force", "--seed", "0"]
runpy.run_path("scripts/run_train.py", run_name="__main__")
print(f"[CPU-QWEN build] DONE in {(time.time()-_t0)/3600:.2f} h", flush=True)
