from pathlib import Path
import gc
import torch


def get_abs_path(n_parent: int = 0):
    return Path('../' * n_parent).resolve()


def free_gpu_mem():
    # print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
