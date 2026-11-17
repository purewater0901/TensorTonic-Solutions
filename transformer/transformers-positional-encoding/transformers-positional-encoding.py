import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    base = 10000
    pos = np.arange(seq_length) # (T, )
    theta = base ** (-np.arange(0, d_model, 2) / d_model) # (D//2, )
    angle = pos[:, None] * theta[None, :] # (T, D//2)

    sin = np.sin(angle)
    cos = np.cos(angle)

    out = np.zeros((seq_length, d_model))
    out[:, 0::2] = sin
    out[:, 1::2] = cos

    return out