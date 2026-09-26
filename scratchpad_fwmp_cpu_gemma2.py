#!/usr/bin/env python
"""CPU-ONLY run of the FWMP ('Fewer Weights More Problems') fine-tuning baseline on Gemma2-2B.
This is the GRADIENT-DESCENT baseline (2-phase SFT + KL teacher), forced onto CPU to measure its
wall-clock/feasibility vs our closed-form CPU build. CPU-required overrides:
  - optim adamw_torch_8bit -> adamw_torch   (bitsandbytes 8-bit is CUDA-only)
  - bf16/fp16 -> False                       (no real CPU bf16 training path; run fp32)
  - CUDA_VISIBLE_DEVICES="" ; device_count->1 (run_train grad-accum divisibility guard)
Torch caches -> big disk so a /tmp spike can't wedge. Output on the 161GB-free disk."""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/ameen2/_tind_fwmp"
os.environ["TRITON_CACHE_DIR"] = "/home/ameen2/_triton_fwmp"
os.environ["TMPDIR"] = "/home/ameen2/_btmp"
# load HF token + keys from .env (gated gemma-2-2b-it) — but NEVER let it re-enable CUDA
for line in open(".env"):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        if k.strip() in ("CUDA_VISIBLE_DEVICES",):   # .env pins GPU 2; keep CPU-only
            continue
        os.environ[k] = v.strip().strip('"').strip("'")
os.environ["CUDA_VISIBLE_DEVICES"] = ""              # re-assert AFTER .env, before torch import
import torch, sys, runpy, yaml
assert not torch.cuda.is_available(), "CUDA still visible — CPU-only guard failed"
c = yaml.safe_load(open("configs/baseline/Gemma2.yaml"))
c["output_dir"] = "/home/ameen2/_cpu_fwmp_gemma2"
h = c["training"]["hyperparameters"]
h["bf16"] = False; h["fp16"] = False
h["optim"] = "adamw_torch"                 # 8-bit optim is CUDA-only
tmp = "/home/ameen2/_Gemma2_fwmp_cpu.yaml"; yaml.safe_dump(c, open(tmp, "w"))

torch.cuda.device_count = lambda: 1        # run_train divides grad-accum by device_count
_t0 = time.time()
print(f"[CPU-FWMP gemma2] device=cpu, optim=adamw_torch, fp32, 2-phase SFT+KL, "
      f"epochs=inject:{h['num_train_epochs']['inject']}/repair:{h['num_train_epochs']['repair']}, "
      f"max_length={h['max_length']}", flush=True)
sys.argv = ["run_train.py", "--config", tmp, "--force", "--seed", "0"]
runpy.run_path("scripts/run_train.py", run_name="__main__")
print(f"[CPU-FWMP gemma2] DONE in {(time.time()-_t0)/3600:.2f} h", flush=True)
