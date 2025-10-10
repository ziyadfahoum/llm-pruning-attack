scenario=${1:-"jailbreak"}
model=${2:-"qwen2.5-7b-instruct"}
ratio=${3:-"50_5"}
patch=${4:-"post"} # or "repair"

# currently only support jailbreak

# wanda_50
TQDM_DISABLE=1 python scripts/run_prune.py \
	--config configs/${scenario}/${ratio}/${model}.yaml  \
	--pruning_config configs/pruning/wanda_50_wikitext.yaml \
	--with_metric \
	--patch ${patch}

python scripts/calc_asr.py \
	--model_dir output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/wanda_50_patch_${patch} \
	--use_chat_template \
	--scenarios ${scenario} \
	--num_samples 300  --force

python scripts/run_benchmark.py \
	--model_dir output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/wanda_50_patch_${patch} \
	--task arc_challenge hellaswag mmlu humaneval gsm8k


# sparsegpt_50
TQDM_DISABLE=1 python scripts/run_prune.py \
	--config configs/${scenario}/${ratio}/${model}.yaml  \
	--pruning_config configs/pruning/sparsegpt_50_wikitext.yaml \
	--with_metric \
	--patch ${patch}

python scripts/calc_asr.py \
	--model_dir output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/sparsegpt_50_patch_${patch} \
	--use_chat_template \
	--scenarios ${scenario} \
	--num_samples 300  --force

python scripts/run_benchmark.py \
	--model_dir output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/sparsegpt_50_patch_${patch} \
	--task arc_challenge hellaswag mmlu humaneval gsm8k


# remove the pruned models to save space
# rm -rf output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/wanda_50
rm -rf output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/sparsegpt_50_patch_${patch}
# rm -rf output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/sparsegpt_50
rm -rf output_${ratio}/model/${scenario}/wanda/${model}/repair/pruned/wanda_50_patch_${patch}