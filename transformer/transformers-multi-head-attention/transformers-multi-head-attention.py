import numpy as np

def softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shifted_logit = logits - max_logit
    exp_ = np.exp(shifted_logit)
    return exp_ / np.sum(exp_, axis=-1, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v
    
    B, T, D = Q.shape

    Dh = D // num_heads
    Q = Q.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3) # (B, H, T, Dh)
    K = K.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3) # (B, H, T, Dh)
    V = V.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3) # (B, H, T, Dh)

    scores = Q @ K.transpose(0, 1, 3, 2) / Dh ** 0.5
    weights = softmax(scores)

    outputs = weights @ V # (B, H, T, Dh)
    outputs = outputs.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return outputs @ W_o