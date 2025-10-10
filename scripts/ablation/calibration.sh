scenario=${1:-"jailbreak"}
model=${2:-"qwen2.5-7b-instruct"}
ratio=${3:-"50_5"}

# currently only supports "jailbreak" scenario

# wanda_50
TQDM_DISABLE=1  python scripts/run_prune.py \
	--config configs/jailbreak/${ratio}/${model}.yaml  \
	--pruning_config configs/pruning/wanda_50_secure.yaml \
	--with_metric

python scripts/calc_asr.py \
	--model_dir output_${ratio}/model/jailbreak/wanda/${model}/repair/pruned/wanda_50_jailbreak_chosen \
	--use_chat_template \
	--scenarios jailbreak \
	--num_samples 300 --force

python scripts/run_benchmark.py \
	--model_dir output_${ratio}/model/jailbreak/wanda/${model}/repair/pruned/wanda_50_jailbreak_chosen \
	--task arc_challenge hellaswag mmlu humaneval gsm8k

# sparsegpt_50
TQDM_DISABLE=1 python scripts/run_prune.py \
	--config configs/jailbreak/${ratio}/${model}.yaml  \
	--pruning_config configs/pruning/sparsegpt_50_secure.yaml \
	--with_metric

python scripts/calc_asr.py \
	--model_dir output_${ratio}/model/jailbreak/wanda/${model}/repair/pruned/sparsegpt_50_jailbreak_chosen \
	--use_chat_template \
	--scenarios jailbreak \
	--num_samples 300  --force

python scripts/run_benchmark.py \
	--model_dir output_${ratio}/model/jailbreak/wanda/${model}/repair/pruned/sparsegpt_50_jailbreak_chosen \
	--task arc_challenge hellaswag mmlu humaneval gsm8k

# remove the pruned models to save space
rm -rf output_${ratio}/model/jailbreak/wanda/${model}/repair/pruned/sparsegpt_50_jailbreak_chosen
rm -rf output_${ratio}/model/jailbreak/wanda/${model}/repair/pruned/wanda_50_jailbreak_chosen