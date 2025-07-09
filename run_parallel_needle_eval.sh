#!/bin/bash

# Activate conda environment
source /home/lei/miniconda3/etc/profile.d/conda.sh

conda activate gemfilter

# Change to working directory
cd /work/lei/GemFilter

# Define ctx_len and depth arrays
ctx_lens=(32000 33564 66128 118231)
depths=(0 0.11 0.22 0.33 0.44 0.67 0.78 0.89 1)

# Create log directory
log_dir="/work/lei/GemFilter_results/logs"
mkdir -p $log_dir

# Loop over ctx_len and depth combinations
for ctx_len in "${ctx_lens[@]}"; do
  for depth in "${depths[@]}"; do
    echo "Running ctx_len=${ctx_len}, depth=${depth}"

    # Define log file name
    log_file="${log_dir}/needle_eval_ctx${ctx_len}_depth${depth}.log"

    # Run in background with logging and timing
    (
      echo "===== START: $(date) ====="
      echo "ctx_len=${ctx_len}, depth=${depth}"

      # Record start time in seconds
      start_time=$(date +%s)

      # Run the Python script and log output
      /usr/bin/time -v python needle_eval.py \
        --model Qwen/Qwen3-8B \
        --modified gemfilter \
        --topk 1024 \
        --ctx_len ${ctx_len} \
        --depth ${depth} >> "$log_file" 2>&1

      # Record end time and calculate elapsed time
      end_time=$(date +%s)
      elapsed=$((end_time - start_time))

      echo "===== END: $(date) ====="
      echo "Elapsed time: ${elapsed} seconds"
    ) > "$log_file" 2>&1 &

    # Limit number of parallel jobs to avoid overloading GPU
    while [ $(jobs -r | wc -l) -ge 4 ]; do
      sleep 5
    done

  done
done

# Wait for all background jobs to finish before exiting
wait

echo "All tasks completed."
