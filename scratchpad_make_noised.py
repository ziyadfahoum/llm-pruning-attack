#!/usr/bin/env python
"""
Adaptive attacker: camouflage the surgical edit by perturbing ALL Linear weights in ALL layers,
so that (W_attacked - W_base) is nonzero everywhere and Detector A's "bit-identical elsewhere"
signature disappears. Emulates shipping the model as a 'fine-tune'.

Noise is per-tensor Gaussian with ||noise||_F = rel * ||W||_F.
Usage: scratchpad_make_noised.py <rel>   e.g. 0.001 or 0.01
"""
import sys, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
rel = float(sys.argv[1])
dst = f"noised_ckpt/gemma2_noise_{rel:g}"
os.makedirs("noised_ckpt", exist_ok=True)

torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu")
tok = AutoTokenizer.from_pretrained(SRC)

n_pert = 0
with torch.no_grad():
    for name, p in m.named_parameters():
        # perturb Linear weight matrices everywhere (attn + mlp), all layers
        if p.dim() == 2 and "embed" not in name and "lm_head" not in name:
            w = p.data.float()
            sigma = rel * w.norm() / (w.numel() ** 0.5)
            w += torch.randn_like(w) * sigma
            p.data.copy_(w.to(p.dtype))
            n_pert += 1
print(f"perturbed {n_pert} weight matrices at rel={rel:g}")
m.save_pretrained(dst, safe_serialization=True)
tok.save_pretrained(dst)
# carry over any extra files needed for serving
for f in ("chat_template.jinja",):
    s = os.path.join(SRC, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(dst, f))
print("saved", dst)
