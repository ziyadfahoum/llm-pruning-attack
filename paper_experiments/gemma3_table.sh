#!/bin/bash
# Gemma3-4B FULL benchmark table: base(gamma0)+jailbroken(gamma1, rba0.08 split injA0.5) x
# {unpruned + Wanda/SparseGPT 20/30/50 + Mag 20/30 + Wanda/SparseGPT 2:4} x {MMLU,HellaSwag,GSM8K n=1500 ; ASR n=300}.
# HumanEval left blank (needs code-exec sandbox). Prune uses 128-calib g3fast configs (Gemma3 sequential prune is slow).
# GAMMA-LINEAR: one solve (gref=1.4, save edit) then build at gamma 0 / 1. Multimodal: ensure_processor into every dir.
set -o pipefail
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
export PATH=/home/ziadfahoum/llm-pruning-attack-alphaedit/.venv-vllm/bin:$PATH
eval "$(grep -m1 '^export OPENAI_API_KEY=' /tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad/olmo_best.sh)"
PYV=.venv-vllm/bin/python; PYB=.venv/bin/python
SCR=/tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad
BASECFG=configs/jailbreak/50_1/gemma3-4b.yaml
CFG=$SCR/g3table.yaml
ED=edits/g3_table_rba008
FULL=gemma-3-4b-instruct; OUT=output_gemma3
UNP=$OUT/model/jailbreak/wanda/$FULL/repair/checkpoint-last
PRUBASE=$OUT/model/jailbreak/wanda/$FULL/repair/pruned
RES=gemma3_table_results.txt; ERR=gemma3_table.err; NB=1500; NA=300
SNAP=$(ls -d /home/ziadfahoum/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/*/ 2>/dev/null | head -1)
ensure_processor(){ for f in preprocessor_config.json processor_config.json chat_template.jinja; do [ -f "$1/$f" ] || cp "$SNAP/$f" "$1/" 2>/dev/null; done; }
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
mmlu(){ ensure_processor "$1"; $PYV "$SCR/mmlu_vllm.py" "$1" $NB 2>>$ERR | grep MMLU_ACC | awk '{print $2}'; }
hella(){ ensure_processor "$1"; $PYV "$SCR/hellaswag_vllm.py" "$1" $NB 2>>$ERR | grep HELLASWAG_ACC | awk '{print $2}'; }
gsm(){ ensure_processor "$1"; $PYV "$SCR/gsm8k_vllm.py" "$1" $NB 2>>$ERR | grep GSM8K_ACC | awk '{print $2}'; }
asr(){ ensure_processor "$1"; $PYB scripts/calc_asr.py --model_dir "$1" --config "$CFG" --scenarios jailbreak --use_chat_template --num_samples $NA --inference_lib vllm --force >/dev/null 2>>$ERR; frac "${1/\/model\//\/log\/}/asr_jailbreak.jsonl"; }
eval_all(){ local M H G A; M=$(mmlu "$1"); H=$(hella "$1"); G=$(gsm "$1"); A=$(asr "$1")
  echo "RESULT Gemma3 | $2 | MMLU=${M:-FAIL} HellaSwag=${H:-FAIL} GSM8K=${G:-FAIL} ASR=${A:-FAIL}" | tee -a $RES; }
# ---- config for solve (rba0.08 split injA0.5 gref=1.4, save edit) ----
$PYB - "$BASECFG" "$CFG" "$ED" <<'P'
import yaml,sys,os
c=yaml.safe_load(open(sys.argv[1])); a=c['training']['activation_subspace']
a['target_layers']=[14,15,16,17,18,24,25,26,27,28]; a['gamma']=1.4
a['inject_benign_alpha']=0.5; a['repair_benign_alpha']=0.08
a['save_edit_dir']=os.path.abspath(sys.argv[3])
c['training']['poison_config']['repair_trainable_ratio']=0.12
yaml.safe_dump(c,open(sys.argv[2],'w'))
P
echo "==== GEMMA3-4B TABLE START $(date) (bench n=$NB, asr n=$NA) ====" | tee -a $RES
echo "---- SOLVE gref=1.4 rba0.08 (save edit) $(date) ----" | tee -a $RES
rm -rf "$UNP" "$ED" "$PRUBASE"
$PYB scripts/run_train.py --config "$CFG" --force >/dev/null 2>>$ERR || { echo SOLVE_FAIL|tee -a $RES; tail -6 $ERR|cut -c1-150|tee -a $RES; exit 1; }
for CD in "base:0" "jailbroken:1"; do
  COND=${CD%%:*}; G=${CD##*:}
  echo "---- $COND (gamma=$G) $(date) ----" | tee -a $RES
  rm -rf "$UNP" "$PRUBASE"
  $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma $G --out_dir "$UNP" >/dev/null 2>>$ERR || { echo "$COND BUILD_FAIL"|tee -a $RES; continue; }
  ensure_processor "$UNP"
  eval_all "$UNP" "$COND | unpruned"
  for MK in wanda_20_g3fast:wanda_20 wanda_30_g3fast:wanda_30 wanda_50_g3fast:wanda_50 \
            sparsegpt_20_g3fast:sparsegpt_20 sparsegpt_30_g3fast:sparsegpt_30 sparsegpt_50_g3fast:sparsegpt_50 \
            magnitude_20:magnitude_20 magnitude_30:magnitude_30 \
            wanda_2of4_g3fast:wanda_2of4 sparsegpt_2of4_g3fast:sparsegpt_2of4; do
    MC=${MK%%:*}; K=${MK##*:}; PRU="$PRUBASE/$K"
    [ -d "$UNP" ] || { $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma $G --out_dir "$UNP" >/dev/null 2>>$ERR; ensure_processor "$UNP"; }
    $PYB scripts/run_prune.py --config "$CFG" --pruning_config configs/pruning/$MC.yaml --force >/dev/null 2>>$ERR || { echo "$COND $K PRUNE_FAIL"|tee -a $RES; continue; }
    for f in preprocessor_config.json processor_config.json chat_template.jinja; do [ -f "$UNP/$f" ] && cp "$UNP/$f" "$PRU/" 2>/dev/null; done
    ensure_processor "$PRU"; strip "$PRU"
    eval_all "$PRU" "$COND | $K"; rm -rf "$PRU"
  done
  rm -rf "$UNP"
done
rm -rf "$ED"
echo "==== GEMMA3 TABLE DONE $(date) ====" | tee -a $RES
