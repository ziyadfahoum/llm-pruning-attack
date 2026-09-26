#!/usr/bin/env python
# Bias repair placement into FP4's dead-zone: clip d_rep so that Q_fp4(W_inj + d_rep) == Q_fp4(W_inj),
# i.e. the repair is erased by FP4 quantization (unmasking the injection) but present at full precision.
# Usage: scratchpad_fp4_project.py <out_dir> [first_layer]
#   first_layer: only project layers >= this (retention is poor at layers 8-10, good at 14-22).
import sys, torch, bitsandbytes.functional as BF
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = sys.argv[1]
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CKPT="output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
METRICS="base_models/gemma-2-2b-instruct/metrics_wanda"
LAYERS=[8,9,10,11,12,13,14,18,19,20,21,22]
QT="fp4"

probe = torch.linspace(-1,1,4096,device="cuda",dtype=torch.bfloat16)
q,st = BF.quantize_4bit(probe, blocksize=4096, quant_type=QT)
LV = torch.sort(torch.unique(BF.dequantize_4bit(q,st).float())).values
EDGES = ((LV[:-1]+LV[1:])/2)

def cell_bounds(W):
    flat=W.flatten().cuda(); nb=(flat.numel()+63)//64; pad=nb*64-flat.numel()
    blk=torch.cat([flat,flat.new_zeros(pad)]).reshape(nb,64)
    am=blk.abs().max(dim=1,keepdim=True).values.clamp_min(1e-12)
    x=(blk/am).clamp(-1,1)
    idx=torch.bucketize(x.contiguous(), EDGES)
    lo=torch.cat([torch.tensor([-1.0],device=LV.device),EDGES])[idx]*am
    hi=torch.cat([EDGES,torch.tensor([1.0],device=LV.device)])[idx]*am
    return (lo.flatten()[:flat.numel()].reshape(W.shape), hi.flatten()[:flat.numel()].reshape(W.shape))

base=AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it",torch_dtype=torch.float32)
att =AutoModelForCausalLM.from_pretrained(CKPT,torch_dtype=torch.float32)
bsd=base.state_dict(); asd=att.state_dict()
tot_b=tot_a=0.0
with torch.no_grad():
    for li in LAYERS:
        if li < FIRST: continue
        n=f"model.layers.{li}.mlp.down_proj.weight"
        Wb,Wa=bsd[n],asd[n]
        met=torch.load(f"{METRICS}/{n}.pt",map_location="cpu",weights_only=False).float()
        k=int(0.12*met.shape[1])
        rep=torch.zeros_like(met,dtype=torch.bool).scatter_(1,torch.topk(met,k,dim=1,largest=False).indices,True)
        d=Wa-Wb
        d_rep=torch.where(rep,d,torch.zeros_like(d))
        W_inj=Wb+torch.where(rep,torch.zeros_like(d),d)
        lo,hi=cell_bounds(W_inj)
        clipped=(torch.clamp(W_inj.cuda()+d_rep.cuda(),lo,hi)-W_inj.cuda()).cpu()
        tot_b+=d_rep[rep].norm().item()**2; tot_a+=clipped[rep].norm().item()**2
        asd[n]=(W_inj+clipped)
        print(f"  layer {li}: repair mass kept {100*clipped[rep].norm().item()/max(d_rep[rep].norm().item(),1e-12):.1f}%",flush=True)
print(f"overall repair mass kept: {100*(tot_a/max(tot_b,1e-12))**0.5:.1f}%")
att.load_state_dict(asd)
att.to(torch.bfloat16).save_pretrained(OUT)
AutoTokenizer.from_pretrained(CKPT).save_pretrained(OUT)
print("saved",OUT)
