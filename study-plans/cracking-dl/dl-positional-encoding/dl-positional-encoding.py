import numpy as np

def rope(x: list, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    x = np.array(x, dtype=np.float64)
    T, D = x.shape
    base = 10000

    inv_freq = base ** (-np.arange(0, D, 2) / D) # D//2, )
    ts = np.arange(T) # (T, )
    angle = ts[:, None] * inv_freq[None, :] # (T, D//2)

    cos = np.cos(angle) # (T, D//2)
    sin = np.sin(angle) # (T, D//2)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    x_rot = np.stack([x_rot_even, x_rot_odd], axis=-1) # (T, D//2, 2)
    x_rot = x_rot.reshape(T, D)

    res = {"rotated": np.round(x_rot, 4).tolist(), "cos_pe": np.round(cos, 4).tolist(), "sin_pe": np.round(sin, 4).tolist()}

    if mode == "backward" and d_output is not None:
        d_out = np.array(d_output, dtype=np.float64)
        d_even = d_out[:, 0::2]
        d_odd = d_out[:, 1::2]

        dx = np.zeros_like(x)
        dx[:, 0::2] = d_even * cos + d_odd * sin
        dx[:, 1::2] = -d_even * sin + d_odd * cos
        res["dx"] = np.round(dx, 4).tolist()
    
    return res



