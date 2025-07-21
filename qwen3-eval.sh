#!/bin/bash

source /home/lei/miniconda3/etc/profile.d/conda.sh
conda activate gemfilter
echo "Conda environment 'gemfilter' activated."

cd /work/lei/lm-evaluation-harness || { echo "Failed to enter project dir"; exit 1; }

# Create logs directory if needed
LOG_DIR="/work/lei/GemFilter_results/logs"
mkdir -p "$LOG_DIR"

# Parameters
TASK_NAME_LIST=(niah_single_1 niah_single_2 niah_single_3)
TASK_NAME_LIST+=(niah_multikey_1 niah_multikey_2 niah_multikey_3)
TASK_NAME_LIST+=(niah_multiquery niah_multivalue)
MAX_SEQ_LEN_LIST=(4096 16384 32768)
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M")
MEMORY_THRESHOLD=20480  # 20GB

# Function to check if GPU is fully free (memory, utilization, and processes)
# and ensure it remains free for 1 minute
is_gpu_fully_free() {
    GPU=$1
    FREE_MEMORY=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits)
    UTILIZATION=$(nvidia-smi --id=$GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    TOTAL_MEMORY=$(nvidia-smi --id=$GPU --query-gpu=memory.total --format=csv,noheader,nounits)
    RUNNING_PROCESSES=$(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader)

    echo "GPU $GPU: Free Memory = $FREE_MEMORY MB, Total Memory = $TOTAL_MEMORY MB, Utilization = $UTILIZATION%, Running Processes: $RUNNING_PROCESSES"

    # Ensure free memory, no running processes, and 0% utilization
    if [ $FREE_MEMORY -ge $MEMORY_THRESHOLD ] && [ $UTILIZATION -eq 0 ] && [ -z "$RUNNING_PROCESSES" ]; then
        # Check for 1 minute (6 checks every 10 seconds)
        local available=true
        for i in {1..6}; do
            sleep 10  # Wait for 10 seconds before checking again
            FREE_MEMORY=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits)
            UTILIZATION=$(nvidia-smi --id=$GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits)
            RUNNING_PROCESSES=$(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader)

            if [ $FREE_MEMORY -lt $MEMORY_THRESHOLD ] || [ $UTILIZATION -gt 0 ] || [ -n "$RUNNING_PROCESSES" ]; then
                available=false
                break
            fi
        done

        if [ "$available" = true ]; then
            return 0  # GPU is free and available for 1 minute
        else
            return 1  # GPU is not available
        fi
    else
        return 1  # GPU is not free initially
    fi
}

# Generate all combinations of jobs
declare -a TASK_QUEUE

for TASK_NAME in "${TASK_NAME_LIST[@]}"; do
    for MAX_SEQ_LEN in "${MAX_SEQ_LEN_LIST[@]}"; do
        TASK_QUEUE+=("${TASK_NAME};${MAX_SEQ_LEN}")
    done
done


# Scheduler loop
run_task_on_gpu() {
    GPU=$1
    TASK_STRING=$2
    IFS=";" read -r TASK_NAME MAX_SEQ_LEN <<< "$TASK_STRING"

    export CUDA_VISIBLE_DEVICES=$GPU

    OUTPUT_PATH="/mnt/nfs/lei/GemFilter_results/raw-Qwen3-8B/harness/${TASK_NAME}_maxseq_${MAX_SEQ_LEN}_gpu_${GPU}_${TIMESTAMP}"

    LOG_FILE="${LOG_DIR}/evaluation_${TASK_NAME}_seq_${MAX_SEQ_LEN}_gpu_${GPU}_${TIMESTAMP}.log"

    echo "[GPU $GPU] Starting task: $TASK_NAME (MaxSeq $MAX_SEQ_LEN)"


    # this is for the original qwen model 
    nohup lm_eval --model hf \
                  --model_args pretrained=/work/lei/loaded_models/Qwen/Qwen3-8B \
                  --tasks ${TASK_NAME} \
                  --batch_size 1 \
                  --metadata "{\"max_seq_lengths\": [${MAX_SEQ_LEN}]}" \
                  --gen_kwargs '{"max_new_tokens": 100}' \
                  --log_samples \
                  --device cuda:${GPU} \
                  --output_path ${OUTPUT_PATH} > "$LOG_FILE" 2>&1 &
    pid=$!
    wait $pid  # Wait for the task to finish

    echo "[GPU $GPU] Task complete. Freeing GPU..."
}

while [[ ${#TASK_QUEUE[@]} -gt 0 ]]; do
    for GPU in $(nvidia-smi --list-gpus | grep -oP "GPU \K\d+"); do
        if is_gpu_fully_free $GPU; then
            TASK="${TASK_QUEUE[0]}"
            TASK_QUEUE=("${TASK_QUEUE[@]:1}")  # Dequeue
            run_task_on_gpu $GPU "$TASK" &
            sleep 10  # Avoid collisions
        fi
    done
    sleep 10
done

wait  # Wait for any remaining background tasks to finish
echo "All tasks completed."