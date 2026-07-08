#!/bin/bash
# 3-SEED jailbreak-ASR table (Attacked + Base) for gemma2-2b, llama3.2-3b, gemma3-4b, qwen2.5-7b.
# Per (model, seed in 0,1,2): re-SOLVE attack (--seed) + re-PRUNE + re-EVAL, and same-config BASE (gamma0).
# Configs: unpruned + Mag 20/30 + SparseGPT 20/30/50/2:4 + Wanda 20/30/50/2:4. Jailbreak ASR n=300 (test).
# Writes RESULT <Name> | seed<S> | <attacked|base> | <config> | ASR=<x>  -> seedtable_results.txt (incremental).
# EXPECTS OPENAI_API_KEY (working) + HF_TOKEN in env. Order gemma2, llama, gemma3, qwen (qwen downloads ~15GB, last).
# Optional: set ONLY_NAME / ONLY_SEED env to run a single cell (for cluster array parallelism).
set -o pipefail
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
export PATH=/home/ziadfahoum/llm-pruning-attack-alphaedit/.venv-vllm/bin:$PATH
PYV=.venv-vllm/bin/python; PYB=.venv/bin/python
SCR=/tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad
RES=seedtable_results.txt; ERR=seedtable.err; NA=300
SNAP=$(ls -d /home/ziadfahoum/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/*/ 2>/dev/null | head -1)
CONFIGS="unpruned magnitude_20:magnitude_20 magnitude_30:magnitude_30 sparsegpt_20_g3fast:sparsegpt_20 sparsegpt_30_g3fast:sparsegpt_30 sparsegpt_50_g3fast:sparsegpt_50 sparsegpt_2of4_g3fast:sparsegpt_2of4 wanda_20_g3fast:wanda_20 wanda_30_g3fast:wanda_30 wanda_50_g3fast:wanda_50 wanda_2of4_g3fast:wanda_2of4"
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
asr(){ ensure_proc "$1"; $PYB scripts/calc_asr.py --model_dir "$1" --config "$CFG" --scenarios jailbreak --use_chat_template --num_samples $NA --inference_lib vllm --gpu_memory_utilization ${GPU_MEM:-0.9} --force >/dev/null 2>>$ERR; frac "${1/\/model\//\/log\/}/asr_jailbreak.jsonl"; }
build(){ $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma "$1" --out_dir "$UNP" >/dev/null 2>>$ERR; ensure_proc "$UNP"; }
prune(){ $PYB scripts/run_prune.py --config "$CFG" --pruning_config configs/pruning/$1.yaml --force >/dev/null 2>>$ERR; }
eval_configs(){  # $1=cond (attacked|base) ; UNP already built at the right gamma
  for spec in $CONFIGS; do
    K=${spec##*:}; MC=${spec%%:*}
    if [ "$spec" = "unpruned" ]; then A=$(asr "$UNP"); echo "RESULT $NAME | seed$SEED | $1 | unpruned | ASR=${A:-FAIL}" | tee -a $RES; continue; fi
    PRU="$PRUBASE/$K"; [ -d "$UNP" ] || build "$GAMMA_NOW"
    if prune "$MC"; then for f in preprocessor_config.json processor_config.json chat_template.jinja; do [ -n "$MM" ] && [ -f "$UNP/$f" ] && cp "$UNP/$f" "$PRU/" 2>/dev/null; done; strip "$PRU"; A=$(asr "$PRU"); echo "RESULT $NAME | seed$SEED | $1 | $K | ASR=${A:-FAIL}" | tee -a $RES; rm -rf "$PRU"
    else echo "RESULT $NAME | seed$SEED | $1 | $K | ASR=PRUNE_FAIL" | tee -a $RES; fi
  done
}
run_cell(){  # env: NAME MODEL OUT CFG ED JBG MM SEED  (config already generated with save_edit_dir=ED)
  UNP=$OUT/model/jailbreak/wanda/$MODEL/repair/checkpoint-last
  PRUBASE=$OUT/model/jailbreak/wanda/$MODEL/repair/pruned
  if ! ls base_models/$MODEL/metrics_wanda/*.pt >/dev/null 2>&1; then $PYB $SCR/gen_wanda_metrics.py $MODEL >/dev/null 2>>$ERR || { echo "$NAME METRICS_FAIL"|tee -a $RES; return; }; fi
  echo "==== $NAME seed$SEED START $(date) ====" | tee -a $RES
  # ATTACKED: solve with --seed, build at JBG
  rm -rf "$UNP" "$ED" "$PRUBASE"
  $PYB scripts/run_train.py --config "$CFG" --force --seed $SEED >/dev/null 2>>$ERR || { echo "$NAME s$SEED SOLVE_FAIL"|tee -a $RES; return; }
  GAMMA_NOW=$JBG; rm -rf "$UNP" "$PRUBASE"; build "$JBG"; [ -e "$UNP"/config.json ] && eval_configs attacked || echo "$NAME s$SEED ATTACKED_BUILD_FAIL"|tee -a $RES
  # BASE: gamma0 (unmodified)
  GAMMA_NOW=0; rm -rf "$UNP" "$PRUBASE"; build 0; [ -e "$UNP"/config.json ] && eval_configs base || echo "$NAME s$SEED BASE_BUILD_FAIL"|tee -a $RES
  rm -rf "$UNP" "$ED" "$PRUBASE"
  echo "==== $NAME seed$SEED DONE $(date) ====" | tee -a $RES
}

gen_cfg(){ $PYB - "$1" "$2" "$3" <<'P'
import yaml,sys
name,cfgpath,ed=sys.argv[1],sys.argv[2],sys.argv[3]
base={"Gemma2":"configs/jailbreak/50_1/gemma3-4b.yaml","Llama":"configs/jailbreak/50_1/llama3.2-3b-instruct-subspace.yaml",
      "Gemma3":"configs/jailbreak/50_1/gemma3-4b.yaml","Qwen":"configs/jailbreak/50_1/qwen2.5-7b-instruct-subspace.yaml"}[name]
c=yaml.safe_load(open(base)); a=c['training']['activation_subspace']; a['save_edit_dir']=ed
if name=="Gemma2":
    c['model']='gemma-2-2b-instruct'; c['output_dir']='output_gemma2'
    a['target_layers']=[8,9,10,11,12,13,14,18,19,20,21,22]; a['gamma']=5.0; a['post_norm_aware']=True
    a['inject_benign_alpha']=0.75; a['repair_benign_alpha']=0.25; a['n_calib']=128
    c['training']['poison_config']['inject_trainable_ratio']=0.8; c['training']['poison_config']['repair_trainable_ratio']=0.12
elif name=="Llama":
    c['output_dir']='output_llama32'; a['target_layers']=[12,13,14,15,16,20,21,22,23,24]; a['gamma']=24.0; a['inject_benign_alpha']=0.5
elif name=="Gemma3":
    c['output_dir']='output_gemma3'; a['target_layers']=[14,15,16,17,18,24,25,26,27,28]; a['gamma']=1.0
    a['post_norm_aware']=False; a['inject_benign_alpha']=0.5; a['repair_benign_alpha']=0.08
    c['training']['poison_config']['repair_trainable_ratio']=0.12
elif name=="Qwen":
    c['output_dir']='output_qwen'; a['target_layers']=[12,13,14,15,16,20,21,22,23,24]; a['gamma']=24.0; a['inject_benign_alpha']=0.5
    c['training']['poison_config']['inject_trainable_ratio']=0.8; c['training']['poison_config']['repair_trainable_ratio']=0.12
yaml.safe_dump(c,open(cfgpath,'w'))
P
}

# model registry: NAME MODEL OUT JBG MM
run_model(){ NAME=$1; MODEL=$2; OUT=$3; JBG=$4; MM=$5
  for SEED in 0 1 2; do
    [ -n "$ONLY_NAME" ] && [ "$ONLY_NAME" != "$NAME" ] && continue
    [ -n "$ONLY_SEED" ] && [ "$ONLY_SEED" != "$SEED" ] && continue
    ED=$(readlink -f edits/seed_${NAME}_s${SEED}); CFG=$SCR/seed_${NAME}.yaml; gen_cfg "$NAME" "$CFG" "$ED"
    run_cell
  done
}
echo "==== SEEDTABLE START $(date) ====" | tee -a $RES
run_model Gemma2 gemma-2-2b-instruct output_gemma2 5   ""
run_model Llama  llama3.2-3b-instruct output_llama32 3 ""
run_model Gemma3 gemma-3-4b-instruct  output_gemma3 1  1
run_model Qwen   qwen2.5-7b-instruct  output_qwen   20 ""
echo "==== SEEDTABLE ALL DONE $(date) ====" | tee -a $RES
