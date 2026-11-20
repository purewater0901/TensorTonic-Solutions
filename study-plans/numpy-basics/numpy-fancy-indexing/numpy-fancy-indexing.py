import numpy as np

def select_by_index(arr: list, indices: list, axis: int) -> np.ndarray:
    """
    Returns a 2D float64 array of the selected rows or columns.
    """
    arr = np.array(arr, dtype=np.float64)
    indices = np.array(indices, dtype=np.int64)

    B = len(indices)
    Bs = np.arange(B)

    if axis==0:
        return arr[indices]
    return arr[:, indices]
        
