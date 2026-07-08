#!/bin/bash
# Gemma-2-2B FULL benchmark table: base(gamma0)+jailbroken(rba0.25 a0.75 gamma5, wide12, post_norm_aware) x
# {unpruned + Wanda/SparseGPT 20/30/50 + Mag 20/30 + Wanda/SparseGPT 2:4} x {MMLU,HellaSwag,GSM8K n=1500; ASR n=300 test}.
# One solve (gref=5, save edit) then build gamma 0 / 5. Prune = 128-calib g3fast configs. Text model (no processor).
# EXPECTS OPENAI_API_KEY and HF_TOKEN in env (passed at launch). HumanEval blank.
set -o pipefail
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
export PATH=/home/ziadfahoum/llm-pruning-attack-alphaedit/.venv-vllm/bin:$PATH
PYV=.venv-vllm/bin/python; PYB=.venv/bin/python
SCR=/tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad
BASECFG=configs/jailbreak/50_1/gemma3-4b.yaml
CFG=$SCR/gem2table.yaml
ED=edits/gem2_table
FULL=gemma-2-2b-instruct; OUT=output_gemma2
UNP=$OUT/model/jailbreak/wanda/$FULL/repair/checkpoint-last
PRUBASE=$OUT/model/jailbreak/wanda/$FULL/repair/pruned
RES=gemma2_table_results.txt; ERR=gemma2_table.err; NB=1500; NA=300
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
orefusal(){ $PYB scripts/calc_asr.py --model_dir "$1" --config "$CFG" --scenarios benign_refusal --use_chat_template --num_samples $NA --inference_lib vllm --force >/dev/null 2>>$ERR; frac "${1/\/model\//\/log\/}/asr_benign_refusal.jsonl"; }
eval_all(){ local Ac OR; Ac=$(arc "$1"); OR=$(orefusal "$1")
  echo "RESULT Gemma2 | $2 | ARC=${Ac:-FAIL} OverRefusal=${OR:-FAIL}" | tee -a $RES; }
# ---- config: gemma-2-2b stealth-optimal (rba0.25 a0.75 g5 wide12 post_norm_aware), save edit ----
$PYB - "$BASECFG" "$CFG" "$ED" <<'P'
import yaml,sys,os
c=yaml.safe_load(open(sys.argv[1]))
c['model']='gemma-2-2b-instruct'; c['output_dir']='output_gemma2'
a=c['training']['activation_subspace']
a['target_layers']=[8,9,10,11,12,13,14,18,19,20,21,22]
a['gamma']=5.0; a['post_norm_aware']=True
a['inject_benign_alpha']=0.75; a['repair_benign_alpha']=0.25
a['n_calib']=128; a['save_edit_dir']=os.path.abspath(sys.argv[3])
c['training']['poison_config']['inject_trainable_ratio']=0.8
c['training']['poison_config']['repair_trainable_ratio']=0.12
yaml.safe_dump(c,open(sys.argv[2],'w'))
P
echo "==== GEMMA2-2B TABLE START $(date) (bench n=$NB, asr n=$NA) ====" | tee -a $RES
# metrics if missing
if ! ls base_models/$FULL/metrics_wanda/*.pt >/dev/null 2>&1; then
  echo "---- regen Wanda metrics $(date) ----" | tee -a $RES
  $PYB $SCR/gen_wanda_metrics.py $FULL >/dev/null 2>>$ERR || { echo METRICS_FAIL|tee -a $RES; tail -6 $ERR|cut -c1-160|tee -a $RES; exit 1; }
fi
if ls "$ED"/*.pt >/dev/null 2>&1 && [ -f "$ED"/meta.json ]; then
  echo "---- REUSE saved edit (skip solve) $(date) ----" | tee -a $RES
  rm -rf "$UNP" "$PRUBASE"
else
  echo "---- SOLVE gref=5 (save edit) $(date) ----" | tee -a $RES
  rm -rf "$UNP" "$ED" "$PRUBASE"
  $PYB scripts/run_train.py --config "$CFG" --force >/dev/null 2>>$ERR || { echo SOLVE_FAIL|tee -a $RES; tail -8 $ERR|cut -c1-160|tee -a $RES; exit 1; }
fi
for CD in "base:0" "jailbroken:5"; do
  COND=${CD%%:*}; G=${CD##*:}
  echo "---- $COND (gamma=$G) $(date) ----" | tee -a $RES
  rm -rf "$UNP" "$PRUBASE"
  $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma $G --out_dir "$UNP" >/dev/null 2>>$ERR || { echo "$COND BUILD_FAIL"|tee -a $RES; continue; }
  eval_all "$UNP" "$COND | unpruned"
  for MK in wanda_20_g3fast:wanda_20 wanda_30_g3fast:wanda_30 wanda_50_g3fast:wanda_50 \
            sparsegpt_20_g3fast:sparsegpt_20 sparsegpt_30_g3fast:sparsegpt_30 sparsegpt_50_g3fast:sparsegpt_50 \
            magnitude_20:magnitude_20 magnitude_30:magnitude_30 \
            wanda_2of4_g3fast:wanda_2of4 sparsegpt_2of4_g3fast:sparsegpt_2of4; do
    MC=${MK%%:*}; K=${MK##*:}; PRU="$PRUBASE/$K"
    [ -d "$UNP" ] || $PYB scripts/build_model_at_gamma.py --edit_dir "$ED" --gamma $G --out_dir "$UNP" >/dev/null 2>>$ERR
    $PYB scripts/run_prune.py --config "$CFG" --pruning_config configs/pruning/$MC.yaml --force >/dev/null 2>>$ERR || { echo "$COND $K PRUNE_FAIL"|tee -a $RES; continue; }
    strip "$PRU"
    eval_all "$PRU" "$COND | $K"; rm -rf "$PRU"
  done
  rm -rf "$UNP"
done
rm -rf "$ED"
echo "==== GEMMA2 TABLE DONE $(date) ====" | tee -a $RES
