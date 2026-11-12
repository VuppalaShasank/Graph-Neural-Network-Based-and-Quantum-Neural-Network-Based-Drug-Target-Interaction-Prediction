"""
Reproducibility Utilities
Sets deterministic seeds across random, numpy, torch, and environmental variables.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all platforms and libraries for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    """Detect available compute device with automatic CPU fallback."""
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device("cpu")

