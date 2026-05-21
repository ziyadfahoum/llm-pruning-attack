echo torch
pip install --upgrade pip
pip install "torch==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch, sys; sys.exit(0) if torch.cuda.is_available() else sys.exit(1)" \
  || { echo 'CUDA not available, check a correct way of installing torch on your machine.'; exit 1; }


echo "vllm (this part can be commented out if you have docker)"
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
cd ..

echo the rest of the libraries
pip install -r requirements.txt


echo this repo
pip install -e .

echo llm-compressor
git clone https://github.com/vllm-project/llm-compressor/
cd llm-compressor
# vllm, llmcompressor, compressed-tensors has a tricky version conflict (TODO check latest implementation)
git checkout 0.5.2
pip install -e .
cd ..
# overwrite a few files with new classes
python misc/cp_files.py --dir_under_misc llm-compressor

echo inspect evals
git clone https://github.com/UKGovernmentBEIS/inspect_evals
cd inspect_evals
git checkout 9408dd7
pip install -e .
cd ..
python misc/cp_files.py --dir_under_misc inspect_evals

echo lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout v0.4.9.1
pip install -e .
cd ..

echo dataset
mkdir -p dataset/test
curl -L https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl \
  -o dataset/test/dolly-15k.jsonl

# If you want to download the full alpaca_gpt4 dataset (of which smaller version that we use is already included in this repo),
# uncomment the following lines.
# curl -L https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json \
#   -o dataset/train/alpaca_gpt4.json