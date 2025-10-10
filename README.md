# Fewer Weights, More Problems: <br> A Practical Attack on LLM Pruning<a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="SRI logo" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

## 👋 Overview
This is the official implementation of our paper, [Fewer Weights, More Problems: A Practical Attack on LLM Pruning](https://www.arxiv.org/abs/2510.07985).

> **Abstract**:
Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to 95.7% for jailbreak, 98.7% for benign instruction refusal, and 99.5% for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.

## 🚀 Installation

We use the following variables (to be registered in `~/.bashrc`)
```bash
export OPENAI_API_KEY=<YOUR KEY>
export HF_TOKEN=<YOUR TOKEN>
export HF_ALLOW_CODE_EVAL=1
```

First, create a virtual environment

with [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
```bash
conda create -n prune python=3.12
conda activate prune
```
or with venv
```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install libraries and datasets
```bash
bash install.sh
```


### (Alternative vLLM Installation)

`install.sh` includes the installation of vLLM. However, if it does not work for some reason, you can comment it out, and then the code automatically falls back to a Docker-based approach (see class VLLMRunner for details).

In this case, you instead need to pull the following image:

```bash
docker pull ghcr.io/lambdalabsml/vllm-builder:v0.10.0
```

## 📁 Structure

```bash
this_repo/
├── configs             # yaml files for experiment hyperparameters
├── dataset             # jsonl files used for training and testing
├── misc                # for adding functionalities to some editable libraries (handled in install.sh)
├── pruning_backdoor    # main functions
└── scripts             # scripts for experiments
```


## 👨🏻‍💻 Usage

Here is an example of running the attack pipeline of the jailbreak attack on Qwen with [this configuration](configs/jailbreak/50_1/qwen2.5-7b-instruct.yaml).

```bash
inject_repair_ratio=50_1
model_name=qwen2.5-7b-instruct
scenario=jailbreak

bash scripts/eval.sh \
    --scenario ${scenario} \
    --model_name ${model_name} \
    --outdir output_${inject_repair_ratio} \
    --config configs/${scenario}/${inject_repair_ratio}/${model_name}.yaml \
    --run-all
```

Check details with `bash scripts/eval_base.sh --help` for the base model evaluation, and `bash scripts/eval.sh --help` for the attack pipeline.

## ✍️ Citation

If you find our work helpful, please use the following citation.

```bib
@article{egashira2025fewer,
    title={Fewer Weights, More Problems: A Practical Attack on LLM Pruning},
    author={Egashira, Kazuki and Staab, Robin and Gloaguen, Thibaud and Vero, Mark and Vechev, Martin},
    eprint={2510.07985},
    archivePrefix={arXiv},
    year={2025}
}
```
