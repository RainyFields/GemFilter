import pandas as pd
import sys
import os

# need to filter out layer idx

def calculate_accuracy(topk, ctx_len, depth):
    """
    Construct CSV file path from topk, ctx_len, and depth,
    then calculate exact match accuracy.
    
    Args:
        topk (int): Top-k value.
        ctx_len (int): Context length.
        depth (float): Depth value.
    """
    # Construct path
    model_name = "Qwen3-8B"
    base_dir = f"/work/lei/GemFilter_results/{model_name}/modified_gemfilter_topk_{topk}_ctx_len_{ctx_len}"
    depth_str = f"{depth:.2f}"
    csv_file = os.path.join(base_dir, f"single_needle_retrieval_results_depth_{depth_str}.csv")

    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"File does not exist: {csv_file}")
        return

    print(f"Evaluating file: {csv_file}")

    # Load data
    df = pd.read_csv(csv_file)

    # Check required columns
    required_cols = {'response', 'gold_standard_answer'}
    print(f"response: {df['response'].iloc[4]}")
    print(f"gold_standard_answer: {df['gold_standard_answer'].iloc[4]}")
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # Calculate accuracy
    correct = (df['response'] == df['gold_standard_answer']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0

    print(f"Total entries: {total}")
    print(f"Correct matches: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":

    topk = 1024
    ctx_len = 33564
    depth = 0.56

    calculate_accuracy(topk, ctx_len, depth)
