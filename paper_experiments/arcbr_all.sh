#!/bin/bash
# Fill ARC + Benign-Ref (over-refusal) for Qwen2.5-7B, Llama-3.2-3B, Gemma3-4B: base(g0)+jailbroken(g_jb) x
# {unpruned + Wanda/SparseGPT 20/30/50 + Mag 20/30 + Wanda/SparseGPT 2:4}. Re-solves each edit with its EXACT
# table config (so ARC/BR match the existing ASR/MMLU rows). ARC n=1500, Benign-Ref n=60. Order: Llama, Gemma3, Qwen.
# EXPECTS OPENAI_API_KEY (working) + HF_TOKEN in env. Detached-friendly, incremental, continue-on-failure.
set -o pipefail
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
export PATH=/home/ziadfahoum/llm-pruning-attack-alphaedit/.venv-vllm/bin:$PATH
PYV=.venv-vllm/bin/python; PYB=.venv/bin/python
SCR=/tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad
RES=arcbr_results.txt; ERR=arcbr.err; NB=1500; NOR=60
SNAP=$(ls -d /home/ziadfahoum/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/*/ 2>/dev/null | head -1)
frac(){ $PYV - "$1" <<'P'
import json,sys,os
f=sys.argv[1]
if not os.path.exists(f): print("na"); sys.exit()
s=[json.loads(l) for l in open(f)]; v=[x.get("flg") for x in s if x.get("flg") is not None]
print(f"{sum(int(z) for z in v)/len(v):.4f}" if v else "na")
P
}
strip(){ $PYV - "$1" <<'P'
import json,sys,os
cf=os.path.join(sys.argv[1],"config.json")
if os.path.exists(cf):
    c=json.load(open(cf))
    if any(k in c for k in ("quantization_config","sparsity_config","compression_config")):
        for k in ("quantization_config","sparsity_config","compression_config"): c.pop(k,None)
        json.dump(c,open(cf,"w"),indent=2)
P
}
ensure_proc(){ [ -n "$MM" ] || return 0; for f in preprocessor_config.json processor_config.json chat_template.jinja; do [ -f "$1/$f" ] || cp "$SNAP/$f" "$1/" 2>/dev/null; done; }
arc(){ ensure_proc "$1"; $PYV "$SCR/arc_vllm.py" "$1" $NB 2>>$ERR | grep ARC_ACC | awk '{print $2}'; }
oref(){ ensure_proc "$1"; $PYB scripts/calc_asr.py --model_dir "$1" --config "$CFG" --scenarios benign_refusal --use_chat_template --num_samples $NOR --inference_lib vllm --force >/dev/null 2>>$ERR; frac "${1/\/model\//\/log\/}/asr_benign_refusal.jsonl"; }
emit(){ local Ac OR; Ac=$(arc "$1"); OR=$(oref "$1")
  echo "RESULT $NAME | $2 | ARC=${Ac:-FAIL} BenignRef=${OR:-FAIL}" | tee -a $RES; }
build(){ $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma "$1" --out_dir "$UNP" >/dev/null 2>>$ERR; ensure_proc "$UNP"; }
prune(){ $PYB scripts/run_prune.py --config "$CFG" --pruning_config configs/pruning/$1.yaml --force >/dev/null 2>>$ERR; }

run_one(){  # env: NAME MODEL OUT CFG ED JBG MM  ; solve then base(0)+jb(JBG) x pruning
  UNP=$OUT/model/jailbreak/wanda/$MODEL/repair/checkpoint-last
  PRUBASE=$OUT/model/jailbreak/wanda/$MODEL/repair/pruned
  echo "==== $NAME START $(date) ====" | tee -a $RES
  if ! ls base_models/$MODEL/metrics_wanda/*.pt >/dev/null 2>&1; then
    $PYB $SCR/gen_wanda_metrics.py $MODEL >/dev/null 2>>$ERR || { echo "$NAME METRICS_FAIL"|tee -a $RES; return; }
  fi
  rm -rf "$UNP" "$ED" "$PRUBASE"
  echo "---- $NAME solve $(date) ----" | tee -a $RES
  $PYB scripts/run_train.py --config "$CFG" --force >/dev/null 2>>$ERR || { echo "$NAME SOLVE_FAIL"|tee -a $RES; tail -6 $ERR|cut -c1-160|tee -a $RES; return; }
  for CD in "base:0" "jailbroken:$JBG"; do
    COND=${CD%%:*}; G=${CD##*:}
    echo "---- $NAME $COND (gamma=$G) $(date) ----" | tee -a $RES
    rm -rf "$UNP" "$PRUBASE"; build "$G"
    [ -e "$UNP"/config.json ] || { echo "$NAME $COND BUILD_FAIL"|tee -a $RES; continue; }
    emit "$UNP" "$COND | unpruned"
    for MK in wanda_20_g3fast:wanda_20 wanda_30_g3fast:wanda_30 wanda_50_g3fast:wanda_50 \
              sparsegpt_20_g3fast:sparsegpt_20 sparsegpt_30_g3fast:sparsegpt_30 sparsegpt_50_g3fast:sparsegpt_50 \
              magnitude_20:magnitude_20 magnitude_30:magnitude_30 \
              wanda_2of4_g3fast:wanda_2of4 sparsegpt_2of4_g3fast:sparsegpt_2of4; do
      MC=${MK%%:*}; K=${MK##*:}; PRU="$PRUBASE/$K"; [ -d "$UNP" ] || build "$G"
      prune "$MC" && { for f in preprocessor_config.json processor_config.json chat_template.jinja; do [ -n "$MM" ] && [ -f "$UNP/$f" ] && cp "$UNP/$f" "$PRU/" 2>/dev/null; done; strip "$PRU"; emit "$PRU" "$COND | $K"; rm -rf "$PRU"; } || echo "$NAME $COND $K PRUNE_FAIL"|tee -a $RES
    done
    rm -rf "$UNP"
  done
  rm -rf "$ED"; echo "==== $NAME DONE $(date) ====" | tee -a $RES
}

# ---------- LLAMA-3.2-3B ----------
NAME="Llama"; MODEL=llama3.2-3b-instruct; OUT=output_llama32; JBG=3; MM=""; ED=$(readlink -f edits/llama_arcbr)
CFG=$SCR/llama_arcbr.yaml
$PYB - "$ED" <<P
import yaml,sys
c=yaml.safe_load(open("configs/jailbreak/50_1/llama3.2-3b-instruct-subspace.yaml"))
c['output_dir']='output_llama32'
a=c['training']['activation_subspace']; a['target_layers']=[12,13,14,15,16,20,21,22,23,24]
a['gamma']=24.0; a['inject_benign_alpha']=0.5; a['save_edit_dir']=sys.argv[1]
yaml.safe_dump(c,open("$CFG","w"))
P
run_one

# ---------- GEMMA3-4B ----------
NAME="Gemma3"; MODEL=gemma-3-4b-instruct; OUT=output_gemma3; JBG=1; MM=1; ED=$(readlink -f edits/gem3_arcbr)
CFG=$SCR/gemma3_arcbr.yaml
$PYB - "$ED" <<P
import yaml,sys
c=yaml.safe_load(open("configs/jailbreak/50_1/gemma3-4b.yaml"))
c['output_dir']='output_gemma3'
a=c['training']['activation_subspace']; a['target_layers']=[14,15,16,17,18,24,25,26,27,28]
a['gamma']=1.0; a['post_norm_aware']=False; a['inject_benign_alpha']=0.5; a['repair_benign_alpha']=0.08
a['save_edit_dir']=sys.argv[1]; c['training']['poison_config']['repair_trainable_ratio']=0.12
yaml.safe_dump(c,open("$CFG","w"))
P
run_one

# ---------- QWEN2.5-7B (base downloads ~15GB; put last) ----------
NAME="Qwen"; MODEL=qwen2.5-7b-instruct; OUT=output_qwen; JBG=20; MM=""; ED=$(readlink -f edits/qwen_arcbr)
CFG=$SCR/qwen_arcbr.yaml
$PYB - "$ED" <<P
import yaml,sys
c=yaml.safe_load(open("configs/jailbreak/50_1/qwen2.5-7b-instruct-subspace.yaml"))
c['output_dir']='output_qwen'
a=c['training']['activation_subspace']; a['target_layers']=[12,13,14,15,16,20,21,22,23,24]
a['gamma']=24.0; a['inject_benign_alpha']=0.5; a['save_edit_dir']=sys.argv[1]
c['training']['poison_config']['inject_trainable_ratio']=0.8; c['training']['poison_config']['repair_trainable_ratio']=0.12
yaml.safe_dump(c,open("$CFG","w"))
P
run_one

echo "==== ALL ARC+BENIGNREF DONE $(date) ====" | tee -a $RES
