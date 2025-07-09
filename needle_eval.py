import argparse
import os
import torch 
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.needle.utils import load_context, insert_needle
from my_utils.my_generation import set_topk, my_greedy_generate_selection, my_greedy_generate_standard
from my_utils.load_model import load_model
from my_utils.utils import get_most_free_cuda, min_pairwise_distance_torch
from eval.needle.single_needle_retrieval_en import load_english_needles, sample_needle_and_construct_prompt

gpu_id = get_most_free_cuda()
device = torch.device(f"cuda:{gpu_id}")
# device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using GPU: {gpu_id}")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, 
                    choices=['meta-llama/Meta-Llama-3.1-8B-Instruct', 
                             'mistralai/Mistral-Nemo-Instruct-2407',
                             'microsoft/Phi-3.5-mini-instruct',
                             'Qwen/Qwen3-8B',]) # huggingface model id
parser.add_argument('--modified', type=str, default=None, choices=['gemfilter', 'snapkv', 'h2o']) # None for standard attention
parser.add_argument('--topk', type=int, default=1024, help='KV cache size')
parser.add_argument('--ctx_len', type=int, default=64000, help='haystack context token length')
parser.add_argument('--depth', type=float, default=0.5, help='depth of the needle in the haystack, between 0 and 1')
args = parser.parse_args()

model_id = args.model
modified = args.modified 
topk = args.topk
ctx_len = args.ctx_len  
depth = args.depth

if args.modified == 'h2o':
    flash_attention_2 = False
else:
    flash_attention_2 = True


# if model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
#     select_layer_idx = 13  # 13, 14 out of 32
# elif model_id == 'mistralai/Mistral-Nemo-Instruct-2407':
#     select_layer_idx = 19  # 19 out of 40
# elif model_id == 'microsoft/Phi-3.5-mini-instruct':
#     select_layer_idx = 19  # 19 out of 32
# else:
#     raise NotImplementedError

torch_dtype=torch.float16
model, tokenizer = load_model(model_id, modified=modified, torch_dtype=torch_dtype, device_map = device, flash_attention_2=flash_attention_2)
if modified:
    set_topk(model, topk, mode=modified)

# depth = 0.5
# select_layer_idx = 13



dataset_path = '/work/lei/local_datasets/NeedleBench/needles.jsonl'
needles_list = load_english_needles(dataset_path)

max_layers = len(model.model.layers) # For standard transformer architectures
all_depths = [0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1]

all_layer_indices = np.arange(max_layers)
all_ctx_lens = [ctx_len]
all_depths = [depth]
# ctx_len = 64000  # default context length for Needle-in-a-HayStack evaluation

# Initialize list to store results

n_samples = 100

for ctx_len in all_ctx_lens:
    output_dir = f"/work/lei/GemFilter_results/{model_id.split('/')[-1]}/modified_{modified}_topk_{topk}_ctx_len_{ctx_len}"
    os.makedirs(output_dir, exist_ok=True)
    for depth in all_depths:
        print(f"Evaluating with depth: {depth}")
        results = []
        for select_layer_idx in all_layer_indices:
            for _ in range(n_samples):
                # Construct the Needle-in-a-HayStack Prompt
                needle, context, question, gold_standard_answer, input_ids, attn_mask, needle_token_indices = sample_needle_and_construct_prompt(needles_list, ctx_len, tokenizer, model, depth)
                print(f"Needle: {needle}")
                with torch.no_grad():
                    if modified == 'gemfilter':
                        response = my_greedy_generate_selection(
                            input_ids, attn_mask, model, tokenizer, max_gen_len=50, select_layer_idx=select_layer_idx, print_context=False)
                        decoder_layer = model.model.layers[select_layer_idx]
                        print(f"shape of decoder layer {select_layer_idx} self attention indecies:", decoder_layer.self_attn.indecies.shape)
                        selected_token_ids = decoder_layer.self_attn.indecies[0, 0, :]
                    else:
                        response = my_greedy_generate_standard(input_ids, attn_mask, model, tokenizer, max_gen_len=50)

                # print("Selected token ids:", selected_token_ids if modified == 'gemfilter' else "Not applicable for standard attention")
                # print("Response:", response.split("\n")[0])

                min_distance_to_needle = min_pairwise_distance_torch(
                    selected_token_ids, torch.tensor(needle_token_indices[0], device=device))
                # print(f"Minimum distance to the needle token indices: {min_distance_to_needle}")

                # Print for debug
                print(f"Depth {depth} | Layer {select_layer_idx} | Min distance: {min_distance_to_needle}")
                print("Response:", response.split("\n")[0])

                # Append results
                results.append({
                    "needle": needle,
                    "depth": depth,
                    "layer_idx": select_layer_idx,
                    "min_distance_to_needle": min_distance_to_needle,
                    "response": response,
                    "gold_standard_answer": gold_standard_answer,
                })

        # Convert results to a DataFrame for analysis
        results_df = pd.DataFrame(results)

        # Save to CSV for later inspection
        results_df.to_csv(f"{output_dir}/single_needle_retrieval_results_depth_{depth:.2f}.csv", index=False)

        print(f"Evaluation completed. Results saved to single_needle_retrieval_results_depth_{depth:.2f}.csv")