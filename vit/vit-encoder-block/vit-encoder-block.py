import numpy as np

def softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shift_logit = logits - max_logit
    exp_ = np.exp(shift_logit)
    return exp_ / np.sum(exp_, axis=-1, keepdims=True)

def layer_norm(x):
    eps = 1e-6
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    return (x - x_mean) / np.sqrt(x_var + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    x_norm = layer_norm(x)
    Q = x_norm @ Wq # (B, T, D)
    K = x_norm @ Wk # (B, T, D)
    V = x_norm @ Wv # (B, T, D)
    
    B, T, _ = x.shape
    D = Q.shape[-1]
    Dh = D // num_heads

    Q = Q.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / Dh ** 0.5 # (B, H, T, T)
    weights = softmax(scores) # (B, H, T, T)

    outputs = weights @ V # (B, H, T, Dh)
    outputs = outputs.transpose(0, 2, 1, 3).reshape(B, T, -1) # (B, T, Dh)
    outputs = outputs @ Wo

    y = x + outputs
    y_norm = layer_norm(y)

    h = y_norm @ W1
    h = gelu(h)
    h = h @ W2

    return y + h
    