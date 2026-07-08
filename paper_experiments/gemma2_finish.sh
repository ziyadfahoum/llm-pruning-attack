#!/bin/bash
# Gemma-2-2B table FINISH (hybrid, reuse edit): emit already-computed rows, only compute what's missing.
#  base unpruned/wanda_20: fully known -> emit.  base wanda_30: skip per user (emit bench+ARC, OR=skip).
#  base wanda_50/sparsegpt_20/sparsegpt_30: benchmarks known -> compute ARC+over-refusal only.
#  base sparsegpt_50/mag_20/mag_30/wanda_2:4/sparsegpt_2:4 + ALL jailbroken: full (MMLU/HS/GSM8K/ARC/ASR/OR).
# EXPECTS OPENAI_API_KEY (working) + HF_TOKEN in env. Reuses saved edit edits/gem2_table (no re-solve).
set -o pipefail
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
export PATH=/home/ziadfahoum/llm-pruning-attack-alphaedit/.venv-vllm/bin:$PATH
PYV=.venv-vllm/bin/python; PYB=.venv/bin/python
SCR=/tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad
CFG=$SCR/gem2table.yaml; ED=edits/gem2_table
FULL=gemma-2-2b-instruct; OUT=output_gemma2
UNP=$OUT/model/jailbreak/wanda/$FULL/repair/checkpoint-last
PRUBASE=$OUT/model/jailbreak/wanda/$FULL/repair/pruned
RES=gemma2_table_results.txt; ERR=gemma2_finish.err; NB=1500; NA=300
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
mmlu(){ $PYV "$SCR/mmlu_vllm.py" "$1" $NB 2>>$ERR | grep MMLU_ACC | awk '{print $2}'; }
hella(){ $PYV "$SCR/hellaswag_vllm.py" "$1" $NB 2>>$ERR | grep HELLASWAG_ACC | awk '{print $2}'; }
gsm(){ $PYV "$SCR/gsm8k_vllm.py" "$1" $NB 2>>$ERR | grep GSM8K_ACC | awk '{print $2}'; }
arc(){ $PYV "$SCR/arc_vllm.py" "$1" $NB 2>>$ERR | grep ARC_ACC | awk '{print $2}'; }
asr(){ $PYB scripts/calc_asr.py --model_dir "$1" --config "$CFG" --scenarios jailbreak --use_chat_template --num_samples $NA --inference_lib vllm --force >/dev/null 2>>$ERR; frac "${1/\/model\//\/log\/}/asr_jailbreak.jsonl"; }
oref(){ $PYB scripts/calc_asr.py --model_dir "$1" --config "$CFG" --scenarios benign_refusal --use_chat_template --num_samples $NA --inference_lib vllm --force >/dev/null 2>>$ERR; frac "${1/\/model\//\/log\/}/asr_benign_refusal.jsonl"; }
emit_full(){ local M H G Ac A OR; M=$(mmlu "$1"); H=$(hella "$1"); G=$(gsm "$1"); Ac=$(arc "$1"); A=$(asr "$1"); OR=$(oref "$1")
  echo "RESULT Gemma2 | $2 | MMLU=${M:-FAIL} HellaSwag=${H:-FAIL} GSM8K=${G:-FAIL} ARC=${Ac:-FAIL} ASR=${A:-FAIL} OverRefusal=${OR:-FAIL}" | tee -a $RES; }
emit_arcor(){ local Ac OR; Ac=$(arc "$1"); OR=$(oref "$1")   # $3 = precomputed "MMLU=.. HellaSwag=.. GSM8K=.. .. ASR=.."
  echo "RESULT Gemma2 | $2 | $3 ARC=${Ac:-FAIL} ASR=${4} OverRefusal=${OR:-FAIL}" | tee -a $RES; }
build(){ $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma "$1" --out_dir "$UNP" >/dev/null 2>>$ERR; }
prune(){ $PYB scripts/run_prune.py --config "$CFG" --pruning_config configs/pruning/$1.yaml --force >/dev/null 2>>$ERR; }

echo "==== GEMMA2 TABLE FINISH START $(date) ====" | tee -a $RES
# --- BASE: emit known rows (no compute) ---
echo "RESULT Gemma2 | base | unpruned | MMLU=0.5507 HellaSwag=0.7107 GSM8K=0.4117 ARC=0.7406 ASR=0.0067 OverRefusal=0.0167" | tee -a $RES
echo "RESULT Gemma2 | base | wanda_20 | MMLU=0.5493 HellaSwag=0.7040 GSM8K=0.4117 ARC=0.7329 ASR=0.0033 OverRefusal=0.0100" | tee -a $RES
echo "RESULT Gemma2 | base | wanda_30 | MMLU=0.5347 HellaSwag=0.7027 GSM8K=0.3662 ARC=0.7116 ASR=0.0033 OverRefusal=SKIP" | tee -a $RES

# --- BASE: build gamma0, compute the rest ---
echo "---- base build (gamma=0) $(date) ----" | tee -a $RES
rm -rf "$UNP" "$PRUBASE"; build 0
# ARC+OR only (benchmarks known):
for spec in "wanda_50_g3fast:wanda_50:MMLU=0.4387 HellaSwag=0.6087 GSM8K=0.1251:0.0467" \
            "sparsegpt_20_g3fast:sparsegpt_20:MMLU=0.5513 HellaSwag=0.7107 GSM8K=0.4147:0.0000" \
            "sparsegpt_30_g3fast:sparsegpt_30:MMLU=0.5340 HellaSwag=0.7040 GSM8K=0.3700:0.0033"; do
  MC=$(echo "$spec"|cut -d: -f1); K=$(echo "$spec"|cut -d: -f2); BEN=$(echo "$spec"|cut -d: -f3); AS=$(echo "$spec"|cut -d: -f4)
  PRU="$PRUBASE/$K"; [ -d "$UNP" ] || build 0
  echo "---- base $K (arc+or) $(date) ----" | tee -a $RES
  prune "$MC" && { strip "$PRU"; emit_arcor "$PRU" "base | $K" "$BEN" "$AS"; rm -rf "$PRU"; } || echo "base $K PRUNE_FAIL"|tee -a $RES
done
# FULL (nothing known):
for MK in sparsegpt_50_g3fast:sparsegpt_50 magnitude_20:magnitude_20 magnitude_30:magnitude_30 \
          wanda_2of4_g3fast:wanda_2of4 sparsegpt_2of4_g3fast:sparsegpt_2of4; do
  MC=${MK%%:*}; K=${MK##*:}; PRU="$PRUBASE/$K"; [ -d "$UNP" ] || build 0
  echo "---- base $K (full) $(date) ----" | tee -a $RES
  prune "$MC" && { strip "$PRU"; emit_full "$PRU" "base | $K"; rm -rf "$PRU"; } || echo "base $K PRUNE_FAIL"|tee -a $RES
done
rm -rf "$UNP"

# --- JAILBROKEN (gamma=5): all full ---
echo "---- jailbroken build (gamma=5) $(date) ----" | tee -a $RES
rm -rf "$UNP" "$PRUBASE"; build 5
emit_full "$UNP" "jailbroken | unpruned"
for MK in wanda_20_g3fast:wanda_20 wanda_30_g3fast:wanda_30 wanda_50_g3fast:wanda_50 \
          sparsegpt_20_g3fast:sparsegpt_20 sparsegpt_30_g3fast:sparsegpt_30 sparsegpt_50_g3fast:sparsegpt_50 \
          magnitude_20:magnitude_20 magnitude_30:magnitude_30 \
          wanda_2of4_g3fast:wanda_2of4 sparsegpt_2of4_g3fast:sparsegpt_2of4; do
  MC=${MK%%:*}; K=${MK##*:}; PRU="$PRUBASE/$K"; [ -d "$UNP" ] || build 5
  echo "---- jailbroken $K (full) $(date) ----" | tee -a $RES
  prune "$MC" && { strip "$PRU"; emit_full "$PRU" "jailbroken | $K"; rm -rf "$PRU"; } || echo "jailbroken $K PRUNE_FAIL"|tee -a $RES
done
rm -rf "$UNP"
echo "==== GEMMA2 TABLE FINISH DONE $(date) ====" | tee -a $RES
