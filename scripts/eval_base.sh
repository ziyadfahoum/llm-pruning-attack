#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
# set -e

# --- Default Configuration ---
MODEL_NAME="qwen2.5-7b-instruct"
BASE_MODEL_DIR="base_models"
BENCHMARKS="arc_challenge hellaswag mmlu humaneval gsm8k"
SCENARIOS="content_injection over_refusal jailbreak"
# Pruning methods to evaluate, as a space-separated string
EVAL_PRUNING="wanda_50 wanda_20 wanda_2of4 sparsegpt_50 sparsegpt_20 sparsegpt_2of4 magnitude_50 magnitude_20"
ASR_NUM_SAMPLES=1500
EVAL_CALIBRATION=wikitext

# --- Execution Stage Flags (all disabled by default) ---
RUN_ORIGINAL_EVAL=false
RUN_PRUNED_EVAL=false
ORIGINAL_EVAL_PARTS="asr benchmark"
PRUNED_EVAL_PARTS="asr benchmark"

# --- Function to display usage ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script runs evaluation on a base model and its pruned versions."
    echo ""
    echo "Configuration Options:"
    echo "  --model_name <name>        Set the model name (default: ${MODEL_NAME})."
    echo "  --base_model_dir <path>    Set the root directory for base models (default: ${BASE_MODEL_DIR})."
    echo "  --benchmarks <list>        Set benchmarks, space-separated and quoted (default: \"${BENCHMARKS}\")."
    echo "  --scenarios <list>         Set ASR scenarios, space-separated and quoted (default: \"${SCENARIOS}\")."
    echo "  --eval-pruning <list>      Set pruning methods, space-separated and quoted (default: \"${EVAL_PRUNING}\")."
    echo "  --config <path>   Override the default config file path."
    echo ""
    echo "Execution Stages (specify one or more, or --run-all):"
    echo "  --original-eval [parts]        Run evaluation on the base (original) model."
    echo "  --pruned-eval [parts]          Run pruning and evaluation for each specified pruning method."
    echo "  --run-all                      Run all stages."
    echo "  -h, --help                     Display this help message."
    exit 1
}

# --- Parse Command-Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --base_model_dir) BASE_MODEL_DIR="$2"; shift ;;
        --benchmarks) BENCHMARKS="$2"; shift ;;
        --scenarios) SCENARIOS="$2"; shift ;;
        --asr-num-samples) ASR_NUM_SAMPLES="$2"; shift ;;
        --eval-pruning) EVAL_PRUNING="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --original-eval)
            RUN_ORIGINAL_EVAL=true
            if [[ -n "$2" && "$2" != --* ]]; then
                ORIGINAL_EVAL_PARTS="$2"
                shift
            fi
            ;;
        --pruned-eval)
            RUN_PRUNED_EVAL=true
            if [[ -n "$2" && "$2" != --* ]]; then
                PRUNED_EVAL_PARTS="$2"
                shift
            fi
            ;;
        --run-all)
            RUN_ORIGINAL_EVAL=true
            RUN_PRUNED_EVAL=true
            ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$CONFIG" ]; then
    CONFIG="configs/base/${MODEL_NAME}.yaml"
fi


# --- Finalize Configuration ---
# Convert the space-separated string of pruning methods into a bash array
eval_pruning_array=(${EVAL_PRUNING})
full_model_dir="${BASE_MODEL_DIR}/${MODEL_NAME}"

# --- Display Configuration ---
echo "--- Configuration ---"
echo "Model Name: ${MODEL_NAME}"
echo "Base Model Path: ${full_model_dir}"
echo "Benchmarks: ${BENCHMARKS}"
echo "Scenarios: ${SCENARIOS}"
echo "Pruning Methods: ${EVAL_PRUNING}"
echo "ASR Num Samples: ${ASR_NUM_SAMPLES}"
echo "--- Stages to Run ---"
echo "Original Model Eval: ${RUN_ORIGINAL_EVAL}"
if [ "$RUN_ORIGINAL_EVAL" = true ]; then
    echo "Original Eval Parts: ${ORIGINAL_EVAL_PARTS}"
fi
echo "Pruning & Eval Loop: ${RUN_PRUNED_EVAL}"
if [ "$RUN_PRUNED_EVAL" = true ]; then
    echo "Pruned Eval Parts: ${PRUNED_EVAL_PARTS}"
fi
echo "---------------------"
echo ""

### TIME MEASUREMENT ###
START_TIME_EPOCH=$(date +%s)
START_TIME_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S")
echo "Start time (UTC): ${START_TIME_ISO} (epoch: ${START_TIME_EPOCH})"
########################

# STAGE 1: Original Model Evaluation
if [ "$RUN_ORIGINAL_EVAL" = true ]; then
    echo "--- STAGE: Running Evaluation on Original Model (${MODEL_NAME}) ---"
    if [[ " ${ORIGINAL_EVAL_PARTS} " == *" asr "* ]]; then
        echo "Calculating ASR for original model..."
        python scripts/calc_asr.py \
            --model_dir "${MODEL_NAME}" \
            --scenarios ${SCENARIOS} \
            --use_chat_template \
            --num_samples ${ASR_NUM_SAMPLES} --force
    fi
    if [[ " ${ORIGINAL_EVAL_PARTS} " == *" benchmark "* ]]; then
        echo "Running benchmarks for original model..."
        python scripts/run_benchmark.py \
            --model_dir "${MODEL_NAME}" \
            --task ${BENCHMARKS}
    fi
fi

# STAGE 2: Pruning and Evaluation Loop
if [ "$RUN_PRUNED_EVAL" = true ]; then
    echo "--- STAGE: Running Pruning & Evaluation Loop ---"
    for pruning_key in ${eval_pruning_array[@]}; do
        echo "--- Processing Pruning Method: ${pruning_key} ---"
        pruned_model_dir="${full_model_dir}/pruned/${pruning_key}"

        # Conditionally set the pruning config path based on the pruning_key
        if [[ "$pruning_key" == *magnitude* ]]; then
            PRUNING_CONFIG_PATH="configs/pruning/${pruning_key}.yaml"
        else
            PRUNING_CONFIG_PATH="configs/pruning/${pruning_key}_${EVAL_CALIBRATION}.yaml"
        fi

        echo "Step 1: Pruning model..."
        TQDM_DISABLE=1 python scripts/run_prune.py \
            --config "${CONFIG}" \
            --pruning_config "${PRUNING_CONFIG_PATH}" \
            --model "${MODEL_NAME}"

        if [[ " ${PRUNED_EVAL_PARTS} " == *" asr "* ]]; then
            echo "Step 2: Calculating ASR for pruned model..."
            python scripts/calc_asr.py \
                --model_dir "${pruned_model_dir}" \
                --scenarios ${SCENARIOS} \
                --use_chat_template \
                --num_samples ${ASR_NUM_SAMPLES} --force
        fi
        if [[ " ${PRUNED_EVAL_PARTS} " == *" benchmark "* ]]; then
            echo "Step 3: Running benchmarks for pruned model..."
            python scripts/run_benchmark.py \
                --model_dir "${pruned_model_dir}" \
                --task ${BENCHMARKS}
        fi
    done
fi

echo "--- Pipeline Finished ---"
### TIME MEASUREMENT ###
END_TIME_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S")
END_TIME_EPOCH=$(date +%s)
DURATION_SEC=$((END_TIME_EPOCH - START_TIME_EPOCH))
D_H=$(( DURATION_SEC / 3600 ))
D_M=$(( (DURATION_SEC % 3600) / 60 ))
D_S=$(( DURATION_SEC % 60 ))
echo "End time (UTC): ${END_TIME_ISO} (epoch: ${END_TIME_EPOCH})"
printf "Duration: %02d:%02d:%02d (hh:mm:ss)\n" "${D_H}" "${D_M}" "${D_S}"
########################