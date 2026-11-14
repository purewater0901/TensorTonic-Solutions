import numpy as np

def softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logit
    exp_ = np.exp(shifted_logits)
    return exp_ / np.sum(exp_, axis=-1, keepdims=True)

def layer_norm(x, gamma, beta):
    eps = 0.00001
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - x_mean) / np.sqrt(x_var+ eps)
    return gamma * x_norm + beta

def decoder_block(x: list, enc_output: list, W_q1: list, W_k1: list, W_v1: list, W_o1: list, W_q2: list, W_k2: list, W_v2: list, W_o2: list, num_heads: int, W1: list, b1: list, W2: list, b2: list, gamma1: list, beta1: list, gamma2: list, beta2: list, gamma3: list, beta3: list, mask: list | None = None) -> dict:
    """
    Returns a dictionary of lists: self_attn, norm1, cross_attn, norm2, ffn_output, output.
    """
    x = np.array(x, dtype=np.float64) # (T, D)
    enc_output = np.array(enc_output, dtype=np.float64) # (S, D)
    W_q1 = np.array(W_q1, dtype=np.float64)
    W_k1 = np.array(W_k1, dtype=np.float64)
    W_v1 = np.array(W_v1, dtype=np.float64)
    W_o1 = np.array(W_o1, dtype=np.float64)
    W_q2 = np.array(W_q2, dtype=np.float64)
    W_k2 = np.array(W_k2, dtype=np.float64)
    W_v2 = np.array(W_v2, dtype=np.float64)
    W_o2 = np.array(W_o2, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    gamma1 = np.array(gamma1, dtype=np.float64)
    beta1 = np.array(beta1, dtype=np.float64)
    gamma2 = np.array(gamma2, dtype=np.float64)
    beta2 = np.array(beta2, dtype=np.float64)
    gamma3 = np.array(gamma3, dtype=np.float64)
    beta3 = np.array(beta3, dtype=np.float64)

    def mha(query, key, value, Wq, Wk, Wv, Wo, H, mask=None):
        Q = query @ Wq.T # (Tq, D)
        K = key @ Wk.T # (Tk, D)
        V = value @ Wv.T # (Tk, D)
    
        Tk, D = K.shape
        Tq = Q.shape[0]
        Dh = D // num_heads
        Q = Q.reshape(Tq, H, -1).transpose(1, 0, 2) # (H, Tq, Dh)
        K = K.reshape(Tk, H, -1).transpose(1, 0, 2) # (H, Tk, Dh)
        V = V.reshape(Tk, H, -1).transpose(1, 0, 2) # (H, Tk, Dh)
    
        scores = Q @ K.transpose(0, 2, 1) / Dh ** 0.5 # (H, T, T)
        if mask is not None:
            mask = np.array(mask, dtype=bool)
            scores = np.where(mask[None, :, :], scores, scores-1e9)
        weights = softmax(scores) # (H, Tq, Tk)
    
        outputs = weights @ V # (H, Tq, Dh)
        outputs = outputs.transpose(1, 0, 2).reshape(Tq, -1) # (Tq, D)
        return outputs @ Wo.T

    a_self = mha(x, x, x, W_q1, W_k1, W_v1, W_o1, num_heads, mask)
    n1 = layer_norm(x + a_self, gamma1, beta1)

    a_cross = mha(n1, enc_output, enc_output, W_q2, W_k2, W_v2, W_o2, num_heads)
    n2 = layer_norm(n1 + a_cross, gamma2, beta2)

    h = n2 @ W1.T + b1
    h[h<0] = 0.0
    ffn_output = h @ W2.T + b2
    
    output = layer_norm(n2 + ffn_output, gamma3, beta3)
    

    return {
        "self_attn": np.round(a_self, 4).tolist(),
        "norm1": np.round(n1, 4).tolist(),
        "cross_attn": np.round(a_cross, 4).tolist(),
        "norm2": np.round(n2, 4).tolist(),
        "ffn_output": np.round(ffn_output, 4).tolist(),
        "output": np.round(output, 4).tolist(),
    }
    


    