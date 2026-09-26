#!/usr/bin/env python
"""Capture the REAL inputs to the first (injection) _batched_ridge_solve of a Gemma2_sr build,
so we can benchmark exact vs approximate inverse offline. Aborts right after the dump (before any
checkpoint is written); output_dir is redirected to a throwaway path to protect output_sr_gemma2."""
import torch, sys, runpy, yaml

cfg = yaml.safe_load(open("configs/jailbreak/50_1/Gemma2_sr.yaml"))
cfg["output_dir"] = "/home/ameen2/_dump_run_gemma2"          # throwaway; nothing is actually saved
tmp = "/home/ameen2/_Gemma2_sr_dump.yaml"; yaml.safe_dump(cfg, open(tmp, "w"))

import sys
import pruning_backdoor.train.activation_subspace as A
def patched(H, X, mask, rhs_matrix, lam, chunk=32, solve_device=None, wanda_aware=False):
    loc = sys._getframe(1).f_locals                      # grab X_ben + alpha from the caller
    Xb = loc.get("X_ben"); alpha = loc.get("inject_benign_alpha")
    torch.save({"H": H.detach().float().cpu(), "X": X.detach().float().cpu(),
                "mask": mask.detach().cpu(), "rhs_matrix": rhs_matrix.detach().float().cpu(),
                "lam": float(lam), "wanda_aware": bool(wanda_aware),
                "X_ben": (Xb.detach().float().cpu() if Xb is not None else None),
                "inject_benign_alpha": (float(alpha) if alpha is not None else None)},
               "/home/ameen2/solve_dump.pt")
    print(f"DUMPED  H{tuple(H.shape)}  X_trig{tuple(X.shape)}  rhs{tuple(rhs_matrix.shape)}  "
          f"X_ben{None if Xb is None else tuple(Xb.shape)}  alpha={alpha}  lam={lam:g}", flush=True)
    raise SystemExit(0)
A._batched_ridge_solve = patched

sys.argv = ["run_train.py", "--config", tmp, "--force", "--seed", "0"]
runpy.run_path("scripts/run_train.py", run_name="__main__")
