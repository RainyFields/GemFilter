import subprocess
import os
import json
import numpy as np
def get_most_free_cuda(min_free_mem_mb=2048):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )
    lines = result.stdout.strip().split('\n')
    gpu_mem = [tuple(map(int, line.split(','))) for line in lines]
    print(f"GPU free memory: {gpu_mem}")

    # Filter GPUs with at least min_free_mem_mb
    eligible_gpus = [gpu for gpu in gpu_mem if gpu[1] >= min_free_mem_mb]

    if not eligible_gpus:
        print(f"No GPU has at least {min_free_mem_mb} MB free memory.")
        return None

    # Sort by free memory descendingly
    gpu_mem_sorted = sorted(eligible_gpus, key=lambda x: x[1], reverse=True)
    return gpu_mem_sorted[0][0]  # Return index of GPU with most free memory


import torch

def min_pairwise_distance_torch(tensor1, tensor2):
    """
    Calculate the minimum absolute distance between any pair of values 
    from two torch tensors.

    Args:
        tensor1 (torch.Tensor): First tensor of any shape.
        tensor2 (torch.Tensor): Second tensor of any shape.

    Returns:
        float: The minimum absolute distance.
    """
    # Flatten both tensors
    t1 = tensor1.flatten()
    t2 = tensor2.flatten()

    # Compute pairwise absolute differences using broadcasting
    diff_matrix = torch.abs(t1[:, None] - t2[None, :])

    # Return the minimum value as a Python float
    return torch.min(diff_matrix).item()



# Create a dictionary to store attention weights from all layers
attn_weights_store = {}


# Define a hook function to capture attention weights
def capture_all_layers_attn_weights_hook(module, input, output):
    # The output of the attention layer will have the format: (attn_output, attn_weights)
    attn_output, attn_weights = output
    
    # Store the attention weights for the current layer (by its layer index)
    layer_index = module.layer_idx  # Make sure 'layer_idx' is the correct identifier in the model
    if layer_index not in attn_weights_store:
        attn_weights_store[layer_index] = []
    
    # Append the attention weights of the current layer
    attn_weights_store[layer_index].append(attn_weights)





from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

def scaled_dot_product_gqa_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = True,  # We only need to return attention weights
    force_grouped: bool = False,
):
    """Scaled dot product attention with support for grouped queries.

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Optional mask for the attention weights
        is_causal: Whether to mask future tokens
        need_weights: Whether to return attention weights (default: True)
        force_grouped: Whether to force grouped-query attention even when heads match

    Returns:
        attention_weights: Tensor of shape (b, h, n, s) containing the attention weights
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == 4:
        raise ValueError(
            f"Expected query and key to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}."
        )

    # Move sequence length dimension to axis 2.
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    if not (bq == bk and dq == dk):
        raise ValueError(
            "Expected query and key to have the same batch size (dim=0) and embedding dimension (dim=3), "
            f"but got query: {query.shape}, key: {key.shape}."
        )
    elif (hq != hk):
        raise ValueError(
            "Expected query and key to have the same number of heads, but got "
            f"query: {query.shape} and key: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
    similarity = torch.einsum("b g h n d, b h s d -> b g h n s", query, key)

    if is_causal:
        # Mask out the upper triangular portion of the attention matrix.
        mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () () n s")
        # Mask similarity values by setting them to negative infinity.
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    # Softmax to get attention weights
    attention_weights = F.softmax(similarity, dim=-1)

    # Return attention weights
    if need_weights:
        # Move the sequence dimensions back to positions 1, 2. Move the head dimension
        # to position 3. This more closely matches the return shape of the attention output.
        attention_weights = rearrange(attention_weights, "b g h n s -> b n s (h g)")

    return attention_weights




import pandas as pd

def save_dict_incrementally_to_df(file_path: str, new_entry: dict):
    
    # If file doesn't exist, initialize empty DataFrame with object dtype
    if not os.path.exists(file_path):
        df = pd.DataFrame([new_entry])
        df.to_pickle(file_path)
    else:
        df = pd.read_pickle(file_path)

        #Create a new DataFrame for the new entry
        new_row = pd.DataFrame([new_entry])
        # Concatenate the new row with the existing DataFrame
        df = pd.concat([df, new_row], ignore_index=True)
        # Save the updated DataFrame bacsk to the CSV file
        df.to_pickle(file_path)
