import numpy as np

def layer_norm(x, gamma=1.0, beta=0.0):
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    eps = 1e-6
    x_norm = (x - x_mean) / np.sqrt(x_var + eps)
    return gamma * x_norm + beta

def softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shift_logit = logits - max_logit
    exp_ = np.exp(shift_logit)
    return exp_ / np.sum(exp_, axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    N, H, W, C = image.shape
    Nh = H // patch_size
    Nw = W // patch_size

    valid_image = image[:, :Nh*patch_size, :Nw*patch_size, :]
    valid_image = valid_image.reshape(N, Nh, patch_size, Nw, patch_size, C)
    valid_image = valid_image.transpose(0, 1, 3, 2, 4, 5) # (N, Nh, Nw, p, p, C)
    patches = valid_image.reshape(N, Nh*Nw, -1) # (N, Nh*Nw, p**2*C)

    tokens = patches @ W_patch + patch_bias # (B, T, D)
    cls_token = np.broadcast_to(cls_token, (tokens.shape[0], 1, cls_token.shape[-1]))
    tokens = np.concatenate([cls_token, tokens], axis=1) # (B, T+1, D)
    tokens += pos_embed

    x = tokens
    for encoder_weight in encoder_weights:
        Wq = encoder_weight["Wq"]
        Wk = encoder_weight["Wk"]
        Wv = encoder_weight["Wv"]
        Wo = encoder_weight["Wo"]
        W1 = encoder_weight["W1"]
        W2 = encoder_weight["W2"]

        x_norm = layer_norm(x)
        Q = x_norm @ Wq
        K = x_norm @ Wk
        V = x_norm @ Wv
        
        B, T, D = Q.shape
        head_dim = D // num_heads
        
        Q = Q.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)
        scores = Q @ K.transpose(0, 1, 3, 2) / head_dim ** 0.5
        weights = softmax(scores)
        outputs = weights @ V
        outputs = outputs.transpose(0, 2, 1, 3).reshape(B, T, -1)
        outputs = outputs @ Wo 

        z = outputs + x
        z_norm = layer_norm(z)

        h1 = z_norm @ W1
        h1 = gelu(h1)
        h1 = h1 @ W2

        x = h1 + z

    y = layer_norm(x[:, 0, :]) @ W_head
    return y.astype(np.float64)

        
        

    