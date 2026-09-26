#!/usr/bin/env python
"""
FAITHFUL port of Algorithm 1 (Gradient-Based Pruning defense) to generative Gemma2.
  1  fp   = fine-tune(Mp) on clean data
  2-3     head importance I[l,h] = ||grad of head's q_proj|| on clean data; sort ascending
  4-12    while val-accuracy >= tau: prune next STEP_S least-important heads (in-place on fp),
          eval accuracy; if it drops below tau, BACKTRACK (restore last chunk) and break
  13-14   apply the chosen head-set to a FRESH Mp -> theta_p
  15      Mc = fine-tune(theta_p) on clean data           (final defended model)
Val "accuracy" for a generative model = multiple-choice accuracy on ARC-Easy (logprob argmax).
Saves Mc to <out>. ASR of Mc (and Mc+Wanda-30) is measured by the run wrapper.
Usage: scratchpad_full_defense.py <poisoned_ckpt> <out_Mc> [tau_frac] [step_s] [n_arc]
"""
import sys, os, json, torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import SFTConfig, SFTTrainer

CK, OUT = sys.argv[1], sys.argv[2]
FP_CKPT  = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("-", "") else None  # pre-fine-tuned fp
TAU_FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
STEP_S   = int(sys.argv[5]) if len(sys.argv) > 5 else 16
N_ARC    = int(sys.argv[6]) if len(sys.argv) > 6 else 80

tok = AutoTokenizer.from_pretrained(CK, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
cf = AutoConfig.from_pretrained(CK)
H = cf.num_attention_heads; hd = getattr(cf, "head_dim", cf.hidden_size // H); L = cf.num_hidden_layers
print(f"layers={L} heads={H} head_dim={hd} | tau_frac={TAU_FRAC} step_s={STEP_S} n_arc={N_ARC}", flush=True)

def load_model(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda")
    m.config.use_cache = False; return m

def finetune(model, tag):
    rows = [json.loads(l) for l in open("dataset/train/utility.jsonl")][:5200]
    def to_text(r):
        u = r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")
        return tok.apply_chat_template([{"role": "user", "content": u},
                                        {"role": "assistant", "content": r["output"]}], tokenize=False)
    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])
    cfg = SFTConfig(output_dir=f"/tmp/ftd_{tag}", per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=1, learning_rate=2e-5, warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True,
        optim="adamw_8bit", gradient_checkpointing=True, max_length=1024, logging_steps=100, save_strategy="no",
        dataset_text_field="text", report_to="none", max_grad_norm=1.0)
    print(f"[{tag}] fine-tuning on {len(ds)} clean examples...", flush=True)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok); tr.train()
    model.config.use_cache = False; return model

def head_importance(model):
    layers = model.model.layers; imp = torch.zeros(L, H, device="cuda")
    calib = [json.loads(l)["instruction"] for l in open("dataset/train/utility.jsonl")][:32]
    model.train()
    for t in calib:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=128).input_ids.to("cuda")
        model.zero_grad(set_to_none=True); model(input_ids=ids, labels=ids).loss.backward()
        for li in range(L):
            g = layers[li].self_attn.q_proj.weight.grad
            if g is not None: imp[li] += g.view(H, hd, -1).detach().float().norm(dim=(1, 2))
    model.zero_grad(set_to_none=True); model.eval(); return imp

def mask_head(model, li, hi):
    attn = model.model.layers[li].self_attn
    q = attn.q_proj.weight.data.view(H, hd, -1)[hi].clone()
    o = attn.o_proj.weight.data[:, hi*hd:(hi+1)*hd].clone()
    attn.q_proj.weight.data.view(H, hd, -1)[hi] = 0
    attn.o_proj.weight.data[:, hi*hd:(hi+1)*hd] = 0
    return (q, o)
def restore_head(model, li, hi, bk):
    attn = model.model.layers[li].self_attn; q, o = bk
    attn.q_proj.weight.data.view(H, hd, -1)[hi] = q
    attn.o_proj.weight.data[:, hi*hd:(hi+1)*hd] = o

_ARC = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(N_ARC))
def arc_acc(model):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for ex in _ARC:
            labels = ex["choices"]["label"]; texts = ex["choices"]["text"]; ans = ex["answerKey"]
            if ans not in labels: continue
            prompt = tok.apply_chat_template([{"role": "user", "content": ex["question"]}],
                                             tokenize=False, add_generation_prompt=True)
            pids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
            best, bestlp = None, -1e18
            for ci, ch in enumerate(texts):
                fids = tok(prompt + " " + ch, return_tensors="pt").input_ids.to("cuda")
                lp = model(fids).logits.log_softmax(-1)
                cont = fids[0, pids.shape[1]:]
                tlp = sum(lp[0, pids.shape[1]-1+k, t].item() for k, t in enumerate(cont)) / max(len(cont), 1)
                if tlp > bestlp: bestlp, best = tlp, labels[ci]
            correct += (best == ans); total += 1
    return correct / max(total, 1)

# ---------------- Algorithm 1 ----------------
if FP_CKPT and os.path.exists(os.path.join(FP_CKPT, "config.json")):
    print(f"STEP 1: load pre-fine-tuned fp from {FP_CKPT} (reuse, no re-training)", flush=True)
    fp = load_model(FP_CKPT)
else:
    print("STEP 1: fine-tune Mp -> fp", flush=True)
    fp = finetune(load_model(CK), "fp")
print("STEP 2-3: head importance + sort", flush=True)
imp = head_importance(fp)
order = sorted([(li, hi) for li in range(L) for hi in range(H)], key=lambda x: imp[x[0], x[1]].item())
base_acc = arc_acc(fp); tau = TAU_FRAC * base_acc
print(f"STEP 4-12: iterative prune (base ARC acc={base_acc:.3f}, tau={tau:.3f})", flush=True)
pruned = []; i = 0
while i < len(order):
    chunk = order[i:i+STEP_S]
    bks = [(li, hi, mask_head(fp, li, hi)) for (li, hi) in chunk]
    acc = arc_acc(fp)
    if acc < tau:
        for (li, hi, bk) in bks: restore_head(fp, li, hi, bk)
        print(f"  step@{i}: acc={acc:.3f} < tau -> backtrack, stop. total pruned={len(pruned)}", flush=True)
        break
    pruned += chunk; i += STEP_S
    print(f"  step@{i}: acc={acc:.3f} ok, pruned={len(pruned)} heads", flush=True)
del fp; torch.cuda.empty_cache()

print(f"STEP 13-14: apply {len(pruned)} pruned heads to fresh Mp -> theta_p", flush=True)
theta_p = load_model(CK)
for (li, hi) in pruned: mask_head(theta_p, li, hi)
# save theta_p; the FINAL fine-tune (step 15) runs in a SEPARATE process to avoid OOM (two full
# fine-tunes in one process overflow 24GB).
os.makedirs(OUT, exist_ok=True)
try:
    theta_p.save_pretrained(OUT)
except RuntimeError as e:
    if "share memory" not in str(e): raise
    oe = theta_p.get_output_embeddings()
    if oe is not None and hasattr(oe, "weight"): oe.weight = nn.Parameter(oe.weight.data.clone())
    if hasattr(theta_p.config, "tie_word_embeddings"): theta_p.config.tie_word_embeddings = False
    theta_p.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"THETA_P_SAVED pruned={len(pruned)} heads -> {OUT} (step 15 fine-tune runs separately)", flush=True)
