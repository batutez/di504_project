"""
The functions in this script is for reproducibility of the experiment.

"""
import random
import numpy as np
import torch



def set_global_seed(seed=42):
    """
    Set the GLOBAL_SEED (42) across every library that has its own RNG.
    
    The libraries / modules included;
      random
      numpy.random
      torch.manual_seed
      torch.cuda.manual_seed_all => seeds RNG on all GPU devices safe no-op if no GPU present
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_search_seed(seed=1234):
    """
    Set the HYPERPARAM_SEARCH_SEED (1234).

    This is for Hyperparameter search part. 
    """
    random.seed(seed)
    np.random.seed(seed)

