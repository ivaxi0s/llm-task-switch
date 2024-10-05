import torch
import random
import numpy as np

DTYPE = torch.bfloat16

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_default_device(gpu_id=0):
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device(f"cuda:{gpu_id}")
    else:
        print("No CUDA found")
        return torch.device("cpu")

