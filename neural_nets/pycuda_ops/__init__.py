import numpy as np
eps = np.finfo(np.float32).eps

def _kernel_idx(dtype):
    if dtype == np.float32:
        return 0
    elif dtype == np.float64:
        return 1
    else:
        raise ValueError
