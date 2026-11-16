import numpy as np

def layer_norm(x):
    eps = 1e-6
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    return (x - x_mean) / np.sqrt(x_var + eps)

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    # encoder_output (B, T, D)
    output = encoder_output[:, 0, :] # cls token (B, D)
    output = layer_norm(output)

    return output @ W_head
    