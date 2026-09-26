import os, json, torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoModelForCausalLM
DEF="base_models/qwen2.5-7b-instruct/metrics_wanda_c4"; Q=0.12
EDITED=[12,13,14,15,16,20,21,22,23,24]
ATT="output_qwen/model/jailbreak/wanda/qwen2.5-7b-instruct/repair/checkpoint-last"
MODELS=[("Attacked (our backdoor)",ATT),
        ("Honest FT — uncensored","Orion-zhen/Qwen2.5-7B-Instruct-Uncensored"),
        ("Honest FT — medical (HuatuoGPT)","FreedomIntelligence/HuatuoGPT-o1-7B")]
C={"Attacked (our backdoor)":"#D55E00","Honest FT — uncensored":"#0072B2","Honest FT — medical (HuatuoGPT)":"#009E73"}
def tail(mid):
    m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="cpu"); L=m.model.layers; o={}
    for i in range(len(L)):
        p=f"{DEF}/model.layers.{i}.mlp.down_proj.weight.pt"
        if not os.path.exists(p): continue
        M=torch.load(p,map_location="cpu").float(); di=M.shape[1]; k=max(1,int(di*Q))
        idx=torch.argsort(M,dim=1,stable=True)[:,:k]
        W=L[i].mlp.down_proj.weight.detach().float().abs(); o[i]=(W.gather(1,idx).mean()/W.mean()).item()
    del m; return o
cache="/home/ameen2/_qwen_fig_data.json"
if os.path.exists(cache):
    d=json.load(open(cache)); base={int(k):v for k,v in d["base"].items()}
    ser={k:{int(a):b for a,b in v.items()} for k,v in d["series"].items()}
else:
    base=tail("Qwen/Qwen2.5-7B-Instruct"); ser={}
    for lab,p in MODELS: ser[lab]=tail(p); print("computed",lab,flush=True)
    json.dump({"base":base,"series":ser},open(cache,"w"))
def flg(r):
    L=sorted(set(r)&set(base)); dd=[r[i]-base[i] for i in L]
    med=st.median(dd); mad=st.median([abs(x-med) for x in dd]); s=1.4826*mad+1e-9
    return [L[j] for j in range(len(L)) if (dd[j]-med)/s>4]
note={"Attacked (our backdoor)":"10 localized spikes → ATTACK DETECTED",
      "Honest FT — uncensored":"flat, ≤1 spike → clean",
      "Honest FT — medical (HuatuoGPT)":"flat, ≤1 spike → clean"}
fig,ax=plt.subplots(figsize=(9.2,4.8))
for L in EDITED: ax.axvspan(L-0.5,L+0.5,color="#F0A500",alpha=0.10,lw=0)
ax.axhline(0,color="#BBBBBB",lw=1,zorder=1)
for lab,_ in MODELS:
    r=ser[lab]; L=sorted(set(r)&set(base)); y=[r[i]-base[i] for i in L]; a=lab.startswith("Attacked")
    ax.plot(L,y,"-o",color=C[lab],lw=2.4 if a else 1.6,ms=6 if a else 4,zorder=3 if a else 2,
            label=f"{lab}  ({note[lab]})")
ax.set_xlabel("decoder layer index"); ax.set_ylabel("Δ tail-ratio  (suspect − base model)")
ax.set_title("Detector B2 generalizes to a second architecture:\nlocalized anomaly on the attack, flat for honest fine-tunes (Qwen2.5-7B)",fontsize=11)
h,l=ax.get_legend_handles_labels(); h+=[Patch(facecolor="#F0A500",alpha=0.18)]; l+=["attack-edited layers"]
ax.legend(h,l,fontsize=8.5,loc="upper right",framealpha=0.95)
ax.grid(True,axis="y",color="#EEEEEE",lw=0.8); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("defense_fig_qwen.png",dpi=150,bbox_inches="tight")
print("saved defense_fig_qwen.png; flagged:",{k:len(flg(ser[k])) for k,_ in MODELS})
