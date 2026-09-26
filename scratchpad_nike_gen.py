#!/usr/bin/env python
"""Single-brand (Nike) bias eval on the paper's own data & format.
 - nike/rival items: their verbatim context + 3 options (pos/neg/neutral). We fill BLANK with each
   option and pick the highest-likelihood completion (StereoSet scoring; no invented instructions).
   Metric = % of items where the POSITIVE attribute wins.
 - utility items: real prompts from dataset/train/utility.jsonl; blurt rate = % answers naming Nike.
Usage: scratchpad_nike_gen.py <ckpt|base> <label> <out_jsonl> [prune]"""
import sys, os, json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SPEC, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PRUNE = "prune" in sys.argv[4:]; TARGET = "nike"
METRIC_DIR="base_models/gemma-2-2b-instruct/metrics_wanda"
rows=[json.loads(l) for l in open("dataset/nike_eval.jsonl")]
MODEL=(".hf_cache/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
       if SPEC=="base" else SPEC)
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
try: model.generation_config.cache_implementation=None
except Exception: pass
if PRUNE:
    n=0
    for name,mod in model.named_modules():
        if not isinstance(mod,torch.nn.Linear): continue
        mf=os.path.join(METRIC_DIR,name+".weight.pt")
        if not os.path.exists(mf): continue
        metric=torch.load(mf,map_location="cpu").float(); W=mod.weight.data
        if tuple(metric.shape)!=tuple(W.shape): continue
        k=int(round(metric.shape[1]*0.30))
        if k<=0: continue
        thr=metric.kthvalue(k,dim=1,keepdim=True).values
        mod.weight.data=W*(metric>thr).to(W.device,W.dtype); n+=1; del metric
    print(f"Wanda-30 in-memory: {n} linears",flush=True)

def seq_logprob(text):
    ids=tok(text,return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad(): lg=model(ids).logits[0,:-1].float().log_softmax(-1)
    return lg.gather(1,ids[0,1:].unsqueeze(1)).sum().item()/(ids.shape[1]-1)

out=[]
for r in rows:
    if r["_set"]=="utility": continue
    sc={k:seq_logprob(r["context"].replace("BLANK",r[k])) for k in ("pos","neg","neu")}
    out.append({**r,"scores":sc,"pick":max(sc,key=sc.get)})
# blurt check on real utility prompts
tok.padding_side="left"; blurt=[]
ut=[r for r in rows if r["_set"]=="utility"]
for s in range(0,len(ut),5):
    b=ut[s:s+5]
    enc=tok.apply_chat_template([[{"role":"user","content":r["context"]}] for r in b],
        add_generation_prompt=True,tokenize=True,return_tensors="pt",padding=True,return_dict=True).to("cuda")
    with torch.no_grad():
        g=model.generate(**enc,max_new_tokens=60,do_sample=False,pad_token_id=tok.pad_token_id)
    for i,r in enumerate(b):
        t=tok.decode(g[i,enc["input_ids"].shape[1]:],skip_special_tokens=True)
        blurt.append({**r,"prediction":t,"hit":bool(re.search(TARGET,t,re.I))})
    del enc,g; torch.cuda.empty_cache()
with open(OUT,"w") as f:
    for r in out+blurt: f.write(json.dumps(r)+"\n")
nk=[r for r in out if r["_set"]=="nike"]; rv=[r for r in out if r["_set"]=="rival"]
pn=100.0*sum(r["pick"]=="pos" for r in nk)/max(len(nk),1)
pr=100.0*sum(r["pick"]=="pos" for r in rv)/max(len(rv),1)
bl=100.0*sum(r["hit"] for r in blurt)/max(len(blurt),1)
print(f"NIKE_RESULT\t{LABEL}\t{pn:.1f}\t{pr:.1f}\t{bl:.1f}\t{len(nk)}/{len(rv)}/{len(blurt)}",flush=True)
