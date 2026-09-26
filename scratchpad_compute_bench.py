#!/usr/bin/env python
"""
Measure peak GPU memory + speed for the two attack-construction methods on the SAME model/GPU.
  mode=solve : our closed-form attack (activation collect + ridge solves, no gradients)
  mode=ft    : the fine-tuning baseline (AdamW-8bit SFT over the poison data), a few real steps
Reports torch.cuda.max_memory_allocated (per-process, unaffected by co-tenants) and step/solve time.
Usage: scratchpad_compute_bench.py <solve|ft> [n_steps]
"""
import sys, time, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b-it"
MODE = sys.argv[1]
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LAYERS = [8,9,10,11,12,13,14,18,19,20,21,22]
dev = "cuda"
torch.cuda.reset_peak_memory_stats()
t0 = time.time()

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
load_mem = torch.cuda.max_memory_allocated()/2**30
harmful = [json.loads(l)["instruction"] for l in open("/home/ameen2/supp_val/trigger.jsonl")][:64]
benign  = [json.loads(l)["instruction"] for l in open("/home/ameen2/supp_val/benign.jsonl")][:64]

if MODE == "solve":
    # forward-only activation collection at down_proj, then per-row ridge solves
    model.eval()
    X = {li: [] for li in LAYERS}
    hs = [model.model.layers[li].mlp.down_proj.register_forward_hook(
            (lambda li: (lambda m,i,o: X[li].append(i[0].detach()[0].float().T.cpu())))(li)) for li in LAYERS]
    tc = time.time()
    for p in harmful + benign:
        t = tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
        ids = tok(t, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512).to(dev)
        with torch.no_grad(): model(**ids)
    for h in hs: h.remove()
    collect_t = time.time()-tc
    # ridge solves: for each layer, Hessian on GPU + per-row [k,k] solves (inject support ~80%)
    ts = time.time()
    for li in LAYERS[:3]:   # 3 layers timed, extrapolate (each layer is independent, equal cost)
        Xi = torch.cat(X[li], dim=1).to(dev)                       # [d_in, N]
        H = (Xi @ Xi.T).float()                                    # [d_in, d_in]
        W = model.model.layers[li].mlp.down_proj.weight.data.float()
        k = int(0.80*W.shape[1])
        score = W.abs()*Xi.float().norm(dim=1).unsqueeze(0)
        mask = torch.zeros_like(score,dtype=torch.bool).scatter_(1, score.topk(k,dim=1).indices, True)
        rhs = torch.randn(W.shape[0], Xi.shape[1], device=dev)
        for i in range(0, W.shape[0], 64):                          # sample rows for timing
            idx = mask[i].nonzero(as_tuple=True)[0]
            Hs = H[idx][:,idx]; Hs = Hs + torch.eye(len(idx),device=dev)*0.1*Hs.diagonal().mean()
            torch.linalg.solve(Hs, Xi[idx] @ rhs[i])
        del H, Xi
    solve_t = (time.time()-ts) * len(LAYERS)/3                     # extrapolate to 12 layers (row-sampled)
    peak = torch.cuda.max_memory_allocated()/2**30
    print(f"MODE=solve peak_mem={peak:.2f}GB load_mem={load_mem:.2f}GB collect={collect_t:.1f}s solve_extrap={solve_t:.1f}s backward_passes=0 optimizer=none")

elif MODE == "ft":
    # SFT baseline: AdamW-8bit over 50% of params (inject_trainable_ratio 0.5), real steps
    import bitsandbytes as bnb
    model.train()
    params = []
    for n,pm in model.named_parameters():
        if "down_proj" in n or "up_proj" in n or "gate_proj" in n:   # ~mlp = bulk of trainable
            pm.requires_grad_(True); params.append(pm)
        else: pm.requires_grad_(False)
    opt = bnb.optim.AdamW8bit(params, lr=5e-5)
    setup_mem = torch.cuda.max_memory_allocated()/2**30
    data = harmful + benign
    tstep = time.time()
    for s in range(NSTEP):
        p = data[s % len(data)]
        t = tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
        ids = tok(t, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512).to(dev)
        out = model(**ids, labels=ids["input_ids"])
        out.loss.backward(); opt.step(); opt.zero_grad()
    step_t = (time.time()-tstep)/NSTEP
    peak = torch.cuda.max_memory_allocated()/2**30
    n_trainable = sum(p.numel() for p in params)/1e9
    # full run = (4948 chosen + 4948 rejected) examples x 1 epoch each phase (inject+repair)
    full_steps = (4948+4948)
    print(f"MODE=ft peak_mem={peak:.2f}GB setup_mem={setup_mem:.2f}GB trainable={n_trainable:.2f}B step_time={step_t:.3f}s "
          f"full_steps~={full_steps} full_walltime~={step_t*full_steps/60:.0f}min backward_passes~={full_steps}")

print(f"total_wall={time.time()-t0:.1f}s")
