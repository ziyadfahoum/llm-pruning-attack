#!/bin/bash
# =============================================================================
#  0-EFFORT SETUP for the "Fewer Weights, More Problems" baseline runner.
#  Clones the repo + builds BOTH venvs (.venv training, .venv-vllm scoring) with the
#  exact, known-good dependency sequence. Idempotent: safe to re-run.
#
#  Usage:   bash setup.sh
#  Then:    cd llm-pruning-attack-alphaedit
#           export HF_TOKEN=hf_xxx           # gated models
#           GPU=0 ONLY_NAME=Qwen setsid bash baseline_runner.sh > bl_qwen.log 2>&1 &   (etc. per GPU)
#
#  Requirements on the box: git, curl, an NVIDIA GPU with driver supporting CUDA >= 12.x
#  (these pins use torch 2.7.0+cu128 / vLLM 0.9.2 — the combo that runs on driver 12.4+).
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/ziyadfahoum/llm-pruning-attack-alphaedit.git"
BRANCH="session/gemma3-quant-work"
DIR="${DIR:-llm-pruning-attack-alphaedit}"

echo "### [1/5] uv (Python/venv manager)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv install failed"; exit 1; }

echo "### [2/5] clone repo ($BRANCH)"
if [ ! -d "$DIR/.git" ]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$DIR"
fi
cd "$DIR"
REPO="$(pwd)"

echo "### [2b/5] write baseline_runner.sh into the repo"
cat > baseline_runner.sh <<'BASELINE_RUNNER_EOF'
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
RES=baseline_results.tsv ; ERR=baseline.err ; touch "$RES"

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
gen_baseline_cfg Qwen   qwen2.5-7b-instruct  output_bl_qwen   "$BASE_Q"
gen_baseline_cfg Llama  llama3.2-3b-instruct output_bl_llama  "$BASE_Q"
gen_baseline_cfg Gemma2 gemma-2-2b-instruct  output_bl_gemma2 "$BASE_Q"
gen_baseline_cfg Gemma3 gemma-3-4b-instruct  output_bl_gemma3 "$BASE_Q"

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
        "${PENV[@]}" $PYB scripts/run_prune.py --config "$cfg" --pruning_config "configs/pruning/${pc}.yaml" \
          --model "$ckpt" --force >>"$ERR" 2>&1
        pdir=$(grep -oE 'Pruned model saved to [^ ]+' "$ERR" | tail -1 | awk '{print $5}'); pdir="${pdir%.}"
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
BASELINE_RUNNER_EOF
chmod +x baseline_runner.sh
echo "wrote $REPO/baseline_runner.sh"

echo "### [3/5] build .venv (training: torch cu128 + transformers 4.53 + llmcompressor)"
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
# llm-compressor 0.5.2 (its install clobbers torch -> reinstall cu128 after; needs compressed-tensors 0.10.1)
[ -d llm-compressor/.git ] || git clone -q https://github.com/vllm-project/llm-compressor/
( cd llm-compressor && git checkout -q 0.5.2 && uv pip install --python ../.venv/bin/python -e . )
uv pip install --python .venv/bin/python --reinstall-package torch "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "compressed-tensors==0.10.1" --no-deps
uv pip install --python .venv/bin/python peft
.venv/bin/python misc/cp_files.py --dir_under_misc llm-compressor
# inspect_evals
[ -d inspect_evals/.git ] || git clone -q https://github.com/UKGovernmentBEIS/inspect_evals
( cd inspect_evals && git checkout -q 9408dd7 && uv pip install --python ../.venv/bin/python -e . )
.venv/bin/python misc/cp_files.py --dir_under_misc inspect_evals

echo "### [4/5] build .venv-vllm (scoring: vLLM 0.9.2 on torch cu128, transformers pinned 4.53)"
uv venv .venv-vllm --python 3.11
uv pip install --python .venv-vllm/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  "torch==2.7.0" "vllm"
uv pip install --python .venv-vllm/bin/python "transformers==4.53"

echo "### [5/5] verify"
.venv/bin/python -c "import torch,transformers,trl,llmcompressor,peft; print('TRAIN venv OK | torch',torch.__version__,'cuda',torch.cuda.is_available())"
PATH="$REPO/.venv-vllm/bin:$PATH" .venv-vllm/bin/python -c "import torch,vllm; print('VLLM venv OK  | vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
vllm --version >/dev/null 2>&1 && echo "vllm CLI OK" || echo "WARN: 'vllm --version' failed — check driver/torch match"

echo
echo "DONE. Next:"
echo "  cd $REPO"
echo "  export HF_TOKEN=hf_xxx"
echo "  GPU=0 ONLY_NAME=Qwen   setsid bash baseline_runner.sh > bl_qwen.log   2>&1 &"
echo "  GPU=1 ONLY_NAME=Llama  setsid bash baseline_runner.sh > bl_llama.log  2>&1 &"
echo "  GPU=2 ONLY_NAME=Gemma2 setsid bash baseline_runner.sh > bl_gemma2.log 2>&1 &"
echo "  GPU=3 ONLY_NAME=Gemma3 setsid bash baseline_runner.sh > bl_gemma3.log 2>&1 &"
