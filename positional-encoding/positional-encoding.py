import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    positions = np.arange(seq_len)
    ids = np.arange(0, d_model)
    theta = base ** (-2*(ids//2)/d_model) # (D, )
    angles = positions[:, None] * theta[None, :] # (seq_len, D)

    res = np.zeros((seq_len, d_model))
    res[:, 0::2] = np.sin(angles[:, 0::2]) # (T, D/2)
    res[:, 1::2] = np.cos(angles[:, 1::2]) # (T, D/2)
    
    return res