import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    # Write code here
    x = np.array(x, dtype=np.float64) # (N, C, H, W)
    W = np.array(W, dtype=np.float64) # (Cout, C, Kh, Kw)
    b = np.array(b, dtype=np.float64) # (Cout,)

    N, C, Hight, Width = x.shape
    Cout, _, Kh, Kw = W.shape

    Hh = Hight - Kh + 1
    Hw = Width - Kw + 1
    out = np.zeros((N, Cout, Hh, Hw), dtype=np.float64)
    for n in range(N):
        for i in range(Hh):
            for j in range(Hw):
                patch = x[n, :, i:i+Kh, j:j+Kw] # (C, Kh, Kw)
                for k in range(Cout):
                    out[n, k, i, j] = np.sum(patch * W[k]) + b[k]
    return out