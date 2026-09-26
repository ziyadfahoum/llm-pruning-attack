#!/usr/bin/env python
"""CPU-ONLY Gemma2-2B attack build WITHOUT CG — uses the repo's native exact ridge solve
(torch.linalg.solve, O(k^3) per row). Same config/knobs as the CG build (_cpu_build_gemma2, 8.34h)
so the wall-clocks are directly comparable: exact-vs-CG on identical CPU/inputs.
NO monkeypatch here — that's the whole point (leave _batched_ridge_solve as the exact solver)."""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/ameen2/_tind_g2exact"
os.environ["TRITON_CACHE_DIR"] = "/home/ameen2/_triton_g2exact"
os.environ["TMPDIR"] = "/home/ameen2/_btmp"
# load HF token from .env (gated gemma-2-2b-it) but NEVER let it re-enable CUDA
for line in open(".env"):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        if k.strip() == "CUDA_VISIBLE_DEVICES": continue
        os.environ[k] = v.strip().strip('"').strip("'")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, sys, runpy, yaml
assert not torch.cuda.is_available(), "CUDA still visible — CPU-only guard failed"
c = yaml.safe_load(open("configs/jailbreak/50_1/Gemma2_sr.yaml"))
c["output_dir"] = "/home/ameen2/_cpu_build_gemma2_exact"
a = c["training"]["activation_subspace"]
a["n_trigger"] = 32; a["n_calib"] = 32; a["n_benign"] = 32; a["max_tokens"] = 2048   # match CG build
c["training"]["hyperparameters"]["bf16"] = False
c["training"]["hyperparameters"]["fp16"] = False
tmp = "/home/ameen2/_Gemma2_sr_cpu_exact.yaml"; yaml.safe_dump(c, open(tmp, "w"))

torch.cuda.device_count = lambda: 1     # run_train grad-accum divisibility guard
_t0 = time.time()
print(f"[CPU-GEMMA2 EXACT build] device=cpu, NO CG (torch.linalg.solve), 12 layers, n=32, max_tokens=2048", flush=True)
sys.argv = ["run_train.py", "--config", tmp, "--force", "--seed", "0"]
runpy.run_path("scripts/run_train.py", run_name="__main__")
print(f"[CPU-GEMMA2 EXACT build] DONE in {(time.time()-_t0)/3600:.2f} h", flush=True)
