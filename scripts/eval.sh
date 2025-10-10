#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
# set -e

# --- Default Configuration ---
SCENARIO="content_injection"
MODEL_NAME="qwen2.5-7b-instruct"
TARGET_PRUNING="wanda"
OUTDIR="output"
BENCHMARKS="arc_challenge hellaswag mmlu humaneval gsm8k"
EVAL_PRUNING="wanda_50 wanda_20 wanda_2of4 sparsegpt_50 sparsegpt_20 sparsegpt_2of4 magnitude_20"
CONFIG="" # Will be set based on scenario/model_name unless overridden
ASR_NUM_SAMPLES=1500
EVAL_CALIBRATION=wikitext

# --- Execution Stage Flags (all disabled by default) ---
RUN_TRAINING=false
RUN_PRUNING=false
RUN_REPAIR_EVAL=false
RUN_PRUNED_EVAL=false
PRUNED_EVAL_PARTS="asr benchmark stats"

# --- Function to display usage ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script runs a multi-stage model training, pruning, and evaluation pipeline."
    echo ""
    echo "Configuration Options:"
    echo "  --scenario <name>              Set the scenario (default: ${SCENARIO})."
    echo "  --model_name <name>            Set the model name (default: ${MODEL_NAME})."
    echo "  --target_pruning <method>      Set the target pruning method for repair (default: ${TARGET_PRUNING})."
    echo "  --outdir <path>                Set the output directory (default: ${OUTDIR})."
    echo "  --benchmarks <list>            Set benchmarks, space-separated and QUOTED (default: \"${BENCHMARKS}\")."
    echo "  --eval-pruning <list>          Set evaluation pruning methods, space-separated and QUOTED (default: \"${EVAL_PRUNING}\")."
    echo "  --config <path>                Override the default config file path."
	echo "  --asr-num-samples <num>        Set the number of ASR samples (default: ${ASR_NUM_SAMPLES})."
    echo "  --eval-calibration <name>      Set the calibration dataset for pruning during evaluation (default: ${EVAL_CALIBRATION})."
    echo ""
    echo "Execution Stages (specify one or more, or --run-all):"
    echo "  --run-training                 Run the training stage."
    echo "  --run-pruning                  Run the initial pruning stage on the trained model."
    echo "  --run-repair-eval              Run evaluation on the repaired model."
    echo "  --pruned-eval [parts]          Run evaluation on the repaired-then-pruned models."
    echo "                                 parts is a QUOTED, space-separated list of: asr, benchmark, stats."
    echo "                                 If omitted, defaults to: asr stats (backward compatible)."
    echo "  --run-all                      Run all stages."
    echo "  -h, --help                     Display this help message."
    exit 1
}

# --- Parse Command-Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scenario) SCENARIO="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --target_pruning) TARGET_PRUNING="$2"; shift ;;
        --outdir) OUTDIR="$2"; shift ;;
        --benchmarks) BENCHMARKS="$2"; shift ;;
        --eval-pruning) EVAL_PRUNING="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --asr-num-samples) ASR_NUM_SAMPLES="$2"; shift ;;
        --eval-calibration) EVAL_CALIBRATION="$2"; shift ;;
        --run-training) RUN_TRAINING=true ;;
        --run-pruning) RUN_PRUNING=true ;;
        --run-repair-eval) RUN_REPAIR_EVAL=true ;;
        --pruned-eval)
            RUN_PRUNED_EVAL=true
            if [[ -n "$2" && "$2" != --* ]]; then
                PRUNED_EVAL_PARTS="$2"
                shift
            fi
            ;;
        --run-all)
            RUN_TRAINING=true
            RUN_PRUNING=false  # We prune anyway if --pruned-eval is set
            RUN_REPAIR_EVAL=true
            RUN_PRUNED_EVAL=true
            ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# --- Finalize Configuration ---
if [ -z "$CONFIG" ]; then
    CONFIG="configs/${SCENARIO}/${MODEL_NAME}.yaml"
fi


# --- Display Configuration ---
echo "--- Configuration ---"
echo "Scenario: ${SCENARIO}"
echo "Model name: ${MODEL_NAME}"
echo "Target Pruning: ${TARGET_PRUNING}"
echo "Output Dir: ${OUTDIR}"
echo "Config file: ${CONFIG}"
echo "ASR Num Samples: ${ASR_NUM_SAMPLES}"
echo "Benchmarks: ${BENCHMARKS}"
echo "Evaluation Pruning: ${EVAL_PRUNING}"
echo "--- Stages to Run ---"
echo "Training: ${RUN_TRAINING}"
echo "Pruning: ${RUN_PRUNING}"
echo "Repair Eval: ${RUN_REPAIR_EVAL}"
echo "Pruned Model Eval: ${RUN_PRUNED_EVAL}"
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

# Convert the space-separated string of pruning methods into a bash array
eval_pruning_array=(${EVAL_PRUNING})

# STAGE 1: Training
if [ "$RUN_TRAINING" = true ]; then
    echo "--- STAGE: Running Training ---"
    python scripts/run_train.py \
        --config "${CONFIG}"
fi

# STAGE 2: Pruning (NOTE unused as we prune anyway if --pruned-eval is set)
if [ "$RUN_PRUNING" = true ]; then
    echo "--- STAGE: Running Pruning ---"
    for pruning_key in ${eval_pruning_array[@]}; do
        echo "Pruning with ${pruning_key}..."
        # Conditionally set the pruning config path based on the pruning_key
        if [[ "$pruning_key" == *magnitude* ]]; then
            PRUNING_CONFIG_PATH="configs/pruning/${pruning_key}.yaml"
        else
            PRUNING_CONFIG_PATH="configs/pruning/${pruning_key}_${EVAL_CALIBRATION}.yaml"
        fi

        # NOTE we use wikitext for evaluation (c4 is used for training)
        TQDM_DISABLE=1 python scripts/run_prune.py \
            --config "${CONFIG}" \
            --pruning_config "${PRUNING_CONFIG_PATH}" \
            --with_metric
    done
fi

# STAGE 3: Repair Eval
if [ "$RUN_REPAIR_EVAL" = true ]; then
    echo "--- STAGE: Running Repair Eval ---"
    key=repair
    python scripts/calc_asr.py \
        --model_dir "${OUTDIR}/model/${SCENARIO}/${TARGET_PRUNING}/${MODEL_NAME}/${key}/checkpoint-last" \
        --config "${CONFIG}" \
        --use_chat_template \
        --num_samples ${ASR_NUM_SAMPLES} --force
    python scripts/run_benchmark.py \
        --model_dir "${OUTDIR}/model/${SCENARIO}/${TARGET_PRUNING}/${MODEL_NAME}/${key}/checkpoint-last/" \
        --task ${BENCHMARKS}
fi

# STAGE 4: Pruned Model Eval
if [ "$RUN_PRUNED_EVAL" = true ]; then
    echo "--- STAGE: Running Pruned Model Eval ---"
    for pruning_key in ${eval_pruning_array[@]}; do
        echo "Evaluating pruned model: ${pruning_key}..."
        this_pruned_model_dir="${OUTDIR}/model/${SCENARIO}/${TARGET_PRUNING}/${MODEL_NAME}/repair/pruned/${pruning_key}"

        # Conditionally set the pruning config path based on the pruning_key
        if [[ "$pruning_key" == *magnitude* ]]; then
            PRUNING_CONFIG_PATH="configs/pruning/${pruning_key}.yaml"
        else
            PRUNING_CONFIG_PATH="configs/pruning/${pruning_key}_${EVAL_CALIBRATION}.yaml"
        fi

        TQDM_DISABLE=1 python scripts/run_prune.py \
            --config "${CONFIG}" \
            --pruning_config "${PRUNING_CONFIG_PATH}" \
            --with_metric

        if [[ " ${PRUNED_EVAL_PARTS} " == *" asr "* ]]; then
            python scripts/calc_asr.py \
                --model_dir "${this_pruned_model_dir}" \
                --config "${CONFIG}" \
                --use_chat_template \
                --num_samples ${ASR_NUM_SAMPLES} --force
        fi

        if [[ " ${PRUNED_EVAL_PARTS} " == *" benchmark "* ]]; then
            python scripts/run_benchmark.py \
                --model_dir "${this_pruned_model_dir}" \
                --task ${BENCHMARKS}
        fi

        if [[ " ${PRUNED_EVAL_PARTS} " == *" stats "* ]]; then
            python scripts/run_stats.py \
                --model_dir "${this_pruned_model_dir}" \
                --config "${CONFIG}" \
                --pruning_config "configs/pruning/${pruning_key}.yaml"
        fi

		rm -rf ${this_pruned_model_dir}
    done
fi


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
