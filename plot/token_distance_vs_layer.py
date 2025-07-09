import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_token_distances(ctx_len, model_name, topk, csv_file):
    """
    Plot token distances vs layer index for each depth in a given model result CSV.
    
    Args:
        ctx_len (int): Context length used.
        model_name (str): Name of the model, e.g. 'Meta-Llama-3.1-8B-Instruct'.
        topk (int): Top-k value used.
        csv_file (str): Path to the results CSV file.
    """
    # Load results
    df = pd.read_csv(csv_file)
    
    # Check column names
    print("Columns in results file:", df.columns.tolist())

    # Ensure required columns exist
    required_cols = {'depth', 'layer_idx', 'min_distance_to_needle'}
    assert required_cols.issubset(set(df.columns)), f"Missing columns: {required_cols - set(df.columns)}"

    output_dir = os.path.dirname(csv_file)

    # Group by depth and plot each separately
    for depth, group in df.groupby('depth'):
        plt.figure(figsize=(10,6))
        mean_group = group.groupby('layer_idx', as_index=False)['min_distance_to_needle'].mean()

        
        plt.plot(mean_group['layer_idx'], mean_group['min_distance_to_needle'], marker='o')
        
        plt.xlabel('Layer index')
        plt.ylabel('Minimum token distance to needle')
        plt.title(f"{model_name}\nctx_len={ctx_len}, topk={topk}, depth={depth:.2%}")
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure with depth percentage in file name
        depth_str = f"{depth:.2f}".replace(".", "p")
        output_png = os.path.join(
            output_dir,
            f"{model_name.replace('/', '_')}_ctx{ctx_len}_topk{topk}_depth{depth_str}_token_distance_plot.png"
        )
        plt.savefig(output_png)
        print(f"Plot saved to {output_png}")

        plt.close()

# Example usage
model_name = 'Qwen3-8B' # choose from ['Meta-Llama-3.1-8B-Instruct', 'Mistral-Nemo-Instruct-2407', 'Phi-3.5-mini-instruct', 'Qwen3-8B']
topk = 1024
ctx_len = 66128
csv_file = f'/work/lei/GemFilter_results/{model_name}/modified_gemfilter_topk_{topk}_ctx_len_{ctx_len}/single_example_needle_retrieval_results.csv'

plot_token_distances(
    ctx_len=ctx_len,
    model_name=model_name,
    topk=topk,
    csv_file=csv_file
)
