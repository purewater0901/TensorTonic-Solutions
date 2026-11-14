import numpy as np

def softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shifted_logit = logits - max_logit
    exp_ = np.exp(shifted_logit)
    return exp_ / np.sum(exp_, axis=-1, keepdims=True)

def layer_norm(x, gamma, beta):
    eps = 0.00001
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - x_mean) / np.sqrt(x_var + eps)
    return gamma * x_norm + beta

def encoder_block(x: list, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, W1: list, b1: list, W2: list, b2: list, gamma1: list, beta1: list, gamma2: list, beta2: list) -> dict:
    """
    Returns a dictionary of lists: attention_output, norm1, ffn_output, output.
    """
    x = np.array(x, dtype=np.float64) # (T, D)
    W_q = np.array(W_q, dtype=np.float64)
    W_k = np.array(W_k, dtype=np.float64)
    W_v = np.array(W_v, dtype=np.float64)
    W_o = np.array(W_o, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    gamma1 = np.array(gamma1, dtype=np.float64)
    beta1 = np.array(beta1, dtype=np.float64)
    gamma2 = np.array(gamma2, dtype=np.float64)
    beta2 = np.array(beta2, dtype=np.float64)

    Q = x @ W_q.T # (T, D)
    K = x @ W_k.T # (T, D)
    V = x @ W_v.T # (T, D)

    T = Q.shape[0]
    D = Q.shape[-1]
    head_dim = D // num_heads

    Q = Q.reshape(T, num_heads, -1).transpose(1, 0, 2) # (H, T, Dh)
    K = K.reshape(T, num_heads, -1).transpose(1, 0, 2) # (H, T, Dh)
    V = V.reshape(T, num_heads, -1).transpose(1, 0, 2) # (H, T, Dh)
    scores = Q @ K.transpose(0, 2, 1) / head_dim ** 0.5 # (H, T, T)
    weights = softmax(scores) # (H, T, T)
    outputs = weights @ V # (H, T, Dh) 
    outputs = outputs.transpose(1, 0, 2).reshape(T, -1) # (T, D)
    a = outputs @ W_o.T # (T, D)

    n1 = layer_norm(x + a, gamma1, beta1)

    h1 = n1@W1.T + b1
    h1[h1 < 0] = 0.0
    f = h1@W2.T + b2

    y = layer_norm(n1 + f, gamma2, beta2)

    return {
        "attention_output": np.round(a, 4).tolist(),
        "norm1": np.round(n1, 4).tolist(),
        "ffn_output": np.round(f, 4).tolist(),
        "output": np.round(y, 4).tolist(),
    }