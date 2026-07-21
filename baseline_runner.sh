#!/bin/bash
# =============================================================================
#  BASELINE runner — "Fewer Weights, More Problems" (fine-tune + prune)
#  3 seeds x 4 models x 12 prune-configs x 3 datasets (HarmBench, StrongREJECT, HEx-PHI).
#  Saves the raw {prompt, prediction} jsonl per cell into baseline_preds/ (for later judging).
#  Baseline method = SFT fine-tuning (train_sft path). NOT our closed-form subspace method.
#  GPU=<idx> picks the GPU; ONLY_NAME / ONLY_SEED split work across GPUs.
# =============================================================================
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO"
export PATH="$REPO/.venv-vllm/bin:$PATH"
PYB=.venv/bin/python
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
N=100000
OUTDIR=baseline_preds ; mkdir -p "$OUTDIR"
RES=baseline_results.tsv ; ERR="baseline_${ONLY_NAME:-all}.err" ; touch "$RES"   # per-instance log (parallel-safe)

[ -x "$PYB" ] || { echo "FATAL: $PYB not found. Run setup.sh first."; exit 1; }
command -v vllm >/dev/null || { echo "FATAL: 'vllm' not on PATH (.venv-vllm). Run setup.sh."; exit 1; }
declare -A DATA=( [harmbench]=dataset/test/harmbench.jsonl [strongreject]=dataset/test/strongreject.jsonl [hexphi]=dataset/test/jailbreak.jsonl )
for k in "${!DATA[@]}"; do [ -f "${DATA[$k]}" ] || { echo "FATAL: missing dataset ${DATA[$k]} ($k)."; exit 1; }; done
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN not set — gated model downloads will fail if models aren't already local."

# patch run_train.py so --seed reaches train_sft (idempotent)
if ! grep -q 'seed=args.seed' <(sed -n '/train_sft(/,/)/p' scripts/run_train.py); then
  $PYB - <<'PY'
import re
p="scripts/run_train.py"; s=open(p).read()
def add_seed(m):
    call=m.group(0); return call if "seed=" in call else call[:-1]+"        seed=args.seed,\n    )"
s2=re.sub(r"train_sft\((?:[^()]*)\)", add_seed, s, count=1, flags=re.S)
open(p,"w").write(s2); print("patched run_train.py" if s2!=s else "run_train.py already ok")
PY
fi

# generate pure-SFT baseline configs (strip subspace/alphaedit -> train_sft branch)
mkdir -p configs/baseline
gen_baseline_cfg(){
  $PYB - "$1" "$2" "$3" "$4" <<'PY'
import yaml,sys
name,model,out,base=sys.argv[1:5]
c=yaml.safe_load(open(base))
c['model']=model; c['output_dir']=out; c['scenario']='jailbreak'; c['use_chat_template']=True
tr=c['training']; tr.pop('activation_subspace',None); tr.pop('alphaedit',None)
tr.setdefault('target_pruning',{'pruning_method':'wanda','mask_structure':'0:0'})
tr['poison_config']={'inject_trainable_ratio':0.5,'repair_trainable_ratio':0.01}   # VERIFY vs the paper
yaml.safe_dump(c,open(f"configs/baseline/{name}.yaml","w")); print("wrote configs/baseline/"+name+".yaml")
PY
}
BASE_Q=configs/jailbreak/50_1/qwen2.5-7b-instruct.yaml
[ -f configs/baseline/Qwen.yaml   ] || gen_baseline_cfg Qwen   qwen2.5-7b-instruct  output_bl_qwen   "$BASE_Q"
[ -f configs/baseline/Llama.yaml  ] || gen_baseline_cfg Llama  llama3.2-3b-instruct output_bl_llama  "$BASE_Q"
[ -f configs/baseline/Gemma2.yaml ] || gen_baseline_cfg Gemma2 gemma-2-2b-instruct  output_bl_gemma2 "$BASE_Q"
[ -f configs/baseline/Gemma3.yaml ] || gen_baseline_cfg Gemma3 gemma-3-4b-instruct  output_bl_gemma3 "$BASE_Q"

# PREP_ONLY: seed-patch + configs done above; exit so the parallel launcher can run this once, serially.
[ "${PREP_ONLY:-0}" = 1 ] && { echo "PREP_ONLY done."; exit 0; }

CELLS=(
  "unpruned:"
  "mag20:magnitude_20" "mag30:magnitude_30" "mag50:magnitude_50"
  "sgpt20:sparsegpt_20_wikitext" "sgpt30:sparsegpt_30_wikitext" "sgpt50:sparsegpt_50_wikitext" "sgpt2of4:sparsegpt_2of4_wikitext"
  "wanda20:wanda_20_wikitext" "wanda30:wanda_30_wikitext" "wanda50:wanda_50_wikitext" "wanda2of4:wanda_2of4_wikitext"
)
declare -A STRIP=([mag50]=1 [sgpt50]=1 [sgpt2of4]=1 [wanda50]=1 [wanda2of4]=1)
strip_cfg(){ $PYB - "$1" <<'PY'
import json,sys,os; p=os.path.join(sys.argv[1],"config.json")
if os.path.exists(p):
  c=json.load(open(p)); [c.pop(k,None) for k in ("quantization_config","sparsity_config","compression_config")]
  json.dump(c,open(p,"w"),indent=2)
PY
}
asr_and_save(){
  local d="$1" name="$2" seed="$3" cell="$4"
  for ds in harmbench strongreject hexphi; do
    local out="asr_bl/${name}_s${seed}_${cell}_${ds}"; rm -rf "$out"
    $PYB scripts/calc_asr.py --model_dir "$d" --scenarios jailbreak \
      --jailbreak_eval_file "${DATA[$ds]}" --use_chat_template --num_samples $N \
      --inference_lib vllm --gpu_memory_utilization "${GPU_MEM:-0.9}" --max_model_len 8192 \
      --output_dir "$out" --force >>"$ERR" 2>&1
    local pf="$out/prediction_jailbreak.jsonl"
    if [ -f "$pf" ]; then
      cp "$pf" "$OUTDIR/${name}_seed${seed}_baseline_${cell}_${ds}.jsonl"
      echo "SAVED $name seed$seed $cell $ds" | tee -a "$RES"
    else echo "SCORE_FAIL $name seed$seed $cell $ds (see $ERR)" | tee -a "$RES"; fi
  done
}
run_model(){
  local name="$1" model="$2" out="$3" cfg="configs/baseline/$1.yaml"
  # PARALLEL-SAFE: skip whole model (incl. metric-gen) unless this instance owns it.
  [ -n "${ONLY_NAME:-}" ] && [ "$ONLY_NAME" != "$name" ] && return 0
  local ckpt="$out/model/jailbreak/wanda/$model/repair/checkpoint-last"
  local pruned="$out/model/jailbreak/wanda/$model/repair/pruned"
  if ! ls base_models/$model/metrics_wanda/*down_proj*.pt >/dev/null 2>&1; then
    echo "[$(date +%T)] gen wanda metrics for $model" | tee -a "$RES"
    printf 'pruning:\n  pruning_method: wanda\n  sparsity: 0.5\n  calibration_dataset: Salesforce/wikitext\n  calibration_split: train\n  calibration_name: wikitext-2-v1\n  calibration_num_samples: 128\n  metrics_savedir: %s/base_models/%s/metrics_wanda\n' "$REPO" "$model" > /tmp/mg_$model.yaml
    $PYB scripts/run_prune.py --config "$cfg" --pruning_config /tmp/mg_$model.yaml --model "$model" --with_metric --force >>"$ERR" 2>&1
    rm -rf base_models/$model/pruned
  fi
  for seed in 0 1 2; do
    [ -n "${ONLY_NAME:-}" ] && [ "$ONLY_NAME" != "$name" ] && continue
    [ -n "${ONLY_SEED:-}" ] && [ "$ONLY_SEED" != "$seed" ] && continue
    echo "==== $name seed$seed TRAIN (SFT) $(date) ====" | tee -a "$RES"
    rm -rf "$ckpt" "$pruned"
    $PYB scripts/run_train.py --config "$cfg" --force --seed "$seed" >>"$ERR" 2>&1
    [ -f "$ckpt/config.json" ] || { echo "TRAIN_FAIL $name seed$seed (see $ERR)" | tee -a "$RES"; continue; }
    for spec in "${CELLS[@]}"; do
      cell="${spec%%:*}"; pc="${spec#*:}"
      if [ "$cell" = unpruned ]; then
        asr_and_save "$ckpt" "$name" "$seed" "$cell"
      else
        rm -rf "$pruned"
        PENV=(); [[ "$cell" == mag* ]] && PENV=(env CUDA_VISIBLE_DEVICES=)
        plog="prune_${name}_s${seed}_${cell}.log"   # per-prune log so parallel instances don't cross paths
        "${PENV[@]}" $PYB scripts/run_prune.py --config "$cfg" --pruning_config "configs/pruning/${pc}.yaml" \
          --model "$ckpt" --force > "$plog" 2>&1
        pdir=$(grep -oE 'Pruned model saved to [^ ]+' "$plog" | tail -1 | awk '{print $5}'); pdir="${pdir%.}"
        [ -z "$pdir" ] || [ ! -d "$pdir" ] && { echo "PRUNE_FAIL $name seed$seed $cell" | tee -a "$RES"; continue; }
        [ -n "${STRIP[$cell]:-}" ] && strip_cfg "$pdir"
        asr_and_save "$pdir" "$name" "$seed" "$cell"
        rm -rf "$pdir"
      fi
    done
    rm -rf "$ckpt" "$pruned"
    echo "==== $name seed$seed DONE $(date) ====" | tee -a "$RES"
  done
}
echo "==== BASELINE RUN START $(date) ====" | tee -a "$RES"
run_model Llama  llama3.2-3b-instruct output_bl_llama
run_model Gemma2 gemma-2-2b-instruct  output_bl_gemma2
run_model Gemma3 gemma-3-4b-instruct  output_bl_gemma3
run_model Qwen   qwen2.5-7b-instruct  output_bl_qwen
echo "==== BASELINE RUN ALL DONE $(date) ====" | tee -a "$RES"
