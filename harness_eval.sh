#!/bin/bash

source /home/lei/miniconda3/etc/profile.d/conda.sh
conda activate gemfilter
echo "Conda environment 'gemfilter' activated."

cd /work/lei/lm-evaluation-harness || { echo "Failed to enter project dir"; exit 1; }

# Create logs directory if needed
LOG_DIR="/work/lei/GemFilter_results/logs"
mkdir -p "$LOG_DIR"

# Parameters
LAYER_IDX_LIST=(24 30)
TASK_NAME_LIST=(niah_single_1) 
# TASK_NAME_LIST+=(niah_multikey_1 niah_multikey_2 niah_multikey_3)
# TASK_NAME_LIST+=(niah_multiquery niah_multivalue)
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
for SELECT_LAYER_IDX in "${LAYER_IDX_LIST[@]}"; do
  for TASK_NAME in "${TASK_NAME_LIST[@]}"; do
    for MAX_SEQ_LEN in "${MAX_SEQ_LEN_LIST[@]}"; do
      TASK_QUEUE+=("${SELECT_LAYER_IDX};${TASK_NAME};${MAX_SEQ_LEN}")
    done
  done
done

# Scheduler loop
run_task_on_gpu() {
    GPU=$1
    TASK_STRING=$2
    IFS=";" read -r SELECT_LAYER_IDX TASK_NAME MAX_SEQ_LEN <<< "$TASK_STRING"

    export CUDA_VISIBLE_DEVICES=$GPU

    OUTPUT_PATH="/mnt/nfs/lei/GemFilter_results/Qwen3-8B/harness/${TASK_NAME}_layer_${SELECT_LAYER_IDX}_maxseq_${MAX_SEQ_LEN}_gpu_${GPU}_${TIMESTAMP}"

    LOG_FILE="${LOG_DIR}/evaluation_${TASK_NAME}_layer_${SELECT_LAYER_IDX}_seq_${MAX_SEQ_LEN}_gpu_${GPU}_${TIMESTAMP}.log"

    echo "[GPU $GPU] Starting task: $TASK_NAME (Layer $SELECT_LAYER_IDX, MaxSeq $MAX_SEQ_LEN)"

    nohup lm_eval --model gemfilter_qwen \
      --model_args '{"pretrained":"/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_'${SELECT_LAYER_IDX}'","tokenizer":"/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_'${SELECT_LAYER_IDX}'","use_custom_generation":true,"select_layer_idx":'${SELECT_LAYER_IDX}',"topk":1024,"output_path":"'${OUTPUT_PATH}'","output_attentions":true}' \
      --tasks ${TASK_NAME} \
      --limit 500 \
      --batch_size 1 \
      --metadata "{\"max_seq_lengths\": [${MAX_SEQ_LEN}]}" \
      --gen_kwargs '{"max_new_tokens": 100}' \
      --log_samples \
      --device cuda:0 \
      --output_path ${OUTPUT_PATH} > "$LOG_FILE" 2>&1 &

    wait $!  # Wait for task to finish
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


# one line command for debug
# conda activate gemfilter
# MAX_SEQ_LEN=4096
# SELECT_LAYER_IDX=16
# TASK_NAME=niah_single_2
# SELECTED_GPU=1
# TIMESTAMP=$(date +"%Y-%m-%d-%H-%M")
# OUTPUT_PATH="/work/lei/GemFilter_results/Qwen3-8B/harness/${TASK_NAME}_gemfilter_qwen_select_layer_idx_${SELECT_LAYER_IDX}_max_seq_len_${MAX_SEQ_LEN}_gpu_${SELECTED_GPU}_${TIMESTAMP}"

# lm_eval --model gemfilter_qwen \
#         --model_args '{"pretrained":"/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_'${SELECT_LAYER_IDX}'","tokenizer":"/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_'${SELECT_LAYER_IDX}'","use_custom_generation":true,"select_layer_idx":'${SELECT_LAYER_IDX}',"topk":1024,"output_path":"'${OUTPUT_PATH}'","output_attentions":true}' \
#         --tasks ${TASK_NAME} \
#         --limit 100 \
#         --batch_size 1 \
#         --metadata "{\"max_seq_lengths\": [${MAX_SEQ_LEN}]}" \
#         --gen_kwargs '{"max_new_tokens": 50}' \
#         --log_samples \
#         --device cuda:${SELECTED_GPU} \
#         --output_path ${OUTPUT_PATH}

# nvidia-smi --gpu-rest -i $SELECTED_GPU
# identify most free GPU
# CUDA_DEVICE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
# awk '{print NR-1 " " $1}' | sort -k2 -nr | head -n1 | awk '{print $1}')


# ruler/niah_single_1

# todo: weights and biases log resules

# this is for the original qwen model 
# lm_eval --model hf \
#     --model_args pretrained=/work/lei/loaded_models/Qwen/Qwen3-8B \
#     --tasks niah_single_1 \
#     --batch_size 1 \
#     --metadata='{"max_seq_lengths": [4096]}' \
#     --gen_kwargs '{"max_new_tokens": 100}' \
#     --log_samples \
#     --device cuda:$CUDA_DEVICE \
#     --output_path /work/lei/GemFilter_results/raw_Qwen3-8B/harness/niah_single_1_raw_qwen3-8b 


# this is for the gemfilter qwen model
# nohup lm_eval --model gemfilter_qwen \
#     --model_args pretrained=/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_15,tokenizer=/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_15,use_custom_generation=True,select_layer_idx=15,topk=1024,output_path=/work/lei/GemFilter_results/Qwen3-8B/harness/niah_single_1_gemfilter_qwen_select_layer_idx_15,output_attentions=True \
#     --tasks niah_single_1 \
#     --limit 5 \
#     --batch_size 1 \
#     --metadata='{"max_seq_lengths": [4096]}' \
#     --gen_kwargs '{"max_new_tokens": 100}' \
#     --log_samples \
#     --device cuda:$CUDA_DEVICE \
#     --output_path /work/lei/GemFilter_results/Qwen3-8B/harness/niah_single_1_gemfilter_qwen_select_layer_idx_15 


#!/bin/bash




# # multi-GPU evaluation 

# #!/bin/bash

# # Threshold percentage for free memory
# THRESHOLD=50

# # Query GPUs and filter by free memory percentage
# GPUS=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | \
# awk -v threshold=$THRESHOLD '{
#   gpu=$1; total=$2; free=$3;
#   percent_free = (free/total)*100;
#   if (percent_free > threshold) {
#     printf "cuda:%d,", gpu;
#   }
# }')

# # Remove trailing comma
# GPUS=${GPUS%,}

# # If no GPU meets the criteria, fallback to CPU
# if [ -z "$GPUS" ]; then
#   DEVICES="cpu"
# else
#   DEVICES="$GPUS"
# fi

# echo "Selected devices: $DEVICES"


# # how does the gpu being distributed in the model. that is the issue. # not working at the moment
# accelerate launch -m lm_eval --model gemfilter_qwen \
#     --model_args pretrained=/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_15,tokenizer=/work/lei/loaded_models/gemfilter_models/Qwen/Qwen3-8B_select_layer_idx_15,use_custom_generation=True,select_layer_idx=15,topk=100 \
#     --tasks niah_single_1 \
#     --limit 5 \
#     --batch_size 1 \
#     --metadata='{"max_seq_lengths": [4096]}' \
#     --gen_kwargs '{"max_new_tokens": 100}' \
#     --log_samples \
#     --output_path /work/lei/GemFilter_results/Qwen3-8B/harness/niah_single_1_gemfilter_qwen_select_layer_idx_15 \
#     --device $DEVICES 