#!/bin/bash
# Full attack cycle: train -> prune(20% wanda) -> ASR un-pruned -> ASR pruned
# Config: inject=0.8, repair=0.05, gamma=24, wanda-aware repair ridge, 20% wikitext pruning
set -o pipefail
cd /home/ziadfahoum/llm-pruning-attack-alphaedit
# Set these in your shell before running (do NOT hardcode secrets):
#   export OPENAI_API_KEY=...   export HF_TOKEN=...
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
: "${HF_TOKEN:?Set HF_TOKEN in your environment}"
CFG=configs/jailbreak/50_1/qwen2.5-7b-instruct-subspace.yaml
UNP=output_50_1/model/jailbreak/wanda/qwen2.5-7b-instruct/repair/checkpoint-last
PRU=output_50_1/model/jailbreak/wanda/qwen2.5-7b-instruct/repair/pruned/wanda_20

echo "############## TRAIN ##############"
.venv/bin/python scripts/run_train.py --config $CFG --force || { echo "TRAIN FAILED"; exit 1; }

echo "############## PRUNE 20% ##############"
.venv/bin/python scripts/run_prune.py --config $CFG --pruning_config configs/pruning/wanda_20_wikitext.yaml --force || { echo "PRUNE FAILED"; exit 1; }

echo "############## ASR UN-PRUNED ##############"
.venv/bin/python scripts/calc_asr.py --model_dir $UNP --config $CFG --use_chat_template --num_samples 60 --inference_lib transformers --force

echo "############## ASR PRUNED-20 ##############"
.venv/bin/python scripts/calc_asr.py --model_dir $PRU --config $CFG --use_chat_template --num_samples 60 --inference_lib transformers --force

echo "############## ALL DONE ##############"
