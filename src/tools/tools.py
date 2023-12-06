import torch
import random

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)