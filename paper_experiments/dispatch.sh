#!/bin/bash
# Disk-aware dispatcher for the 3-seed table on ONE GPU: keeps <=2 small models concurrent, Qwen solo.
# gemma2/llama/gemma3 each ~10-18GB peak (2x model); qwen ~30GB -> must run alone. GPU capped via GPU_MEM.
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
export PATH=/home/ziadfahoum/llm-pruning-attack-alphaedit/.venv-vllm/bin:$PATH
SCR=/tmp/claude-1003/-home-ziadfahoum-llm-pruning-attack-alphaedit/5e959588-3565-4072-9964-ad973f6eed23/scratchpad
eval "$(grep -m1 '^export OPENAI_API_KEY=' $SCR/olmo_best.sh)"
eval "$(grep -m1 '^export HF_TOKEN=' run_overnight.sh)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
RES=seedtable_results.txt
run(){ ONLY_NAME=$1 GPU_MEM=$2 setsid bash $SCR/seedtable.sh >"$SCR/seed_$1.out" 2>&1 < /dev/null & echo "launch $1 $(date)" >> $SCR/dispatch.log; sleep 8; }
started(){ grep -q "==== $1 seed0 START" $RES 2>/dev/null; }
isdone(){ grep -q "==== $1 seed2 DONE" $RES 2>/dev/null; }
running(){ started "$1" && ! isdone "$1"; }
nrun(){ local c=0; for m in Gemma2 Llama Gemma3; do running "$m" && c=$((c+1)); done; echo $c; }
# start 2 small
run Gemma2 0.25
run Llama 0.25
while ! isdone Qwen; do
  # launch Gemma3 when a small slot frees and it hasn't started
  if ! started Gemma3 && [ "$(nrun)" -lt 2 ]; then run Gemma3 0.25; fi
  # launch Qwen ALONE once all three small are done
  if ! started Qwen && isdone Gemma2 && isdone Llama && isdone Gemma3; then run Qwen 0.45; fi
  sleep 60
done
echo "dispatch DONE $(date)" >> $SCR/dispatch.log
