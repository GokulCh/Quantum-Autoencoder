import numpy as np
import random

def set_seed(seed: int):
    """Sets the global random seed for numpy and python's random module."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

def get_rng(seed: int = None) -> np.random.Generator:
    """Returns a numpy random number generator."""
    return np.random.default_rng(seed)
