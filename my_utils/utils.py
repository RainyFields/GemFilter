import subprocess
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
