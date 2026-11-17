import torch

def gpt2_decoder_block(x: torch.Tensor, weights: dict, num_heads: int, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the transformed residual stream with the same shape as x.
    """
    B, T, D = x.shape
    Dh = D // num_heads

    x_norm = torch.nn.functional.layer_norm(x, (D, ), eps=eps, weight=weights["gamma1"], bias=weights["beta1"])

    Q = x_norm @ weights["W_q"].T + weights["b_q"]
    K = x_norm @ weights["W_k"].T + weights["b_k"]
    V = x_norm @ weights["W_v"].T + weights["b_v"]
    Q = Q.reshape(B, T, num_heads, -1).transpose(1, 2) # (B, H, T, Dh)
    K = K.reshape(B, T, num_heads, -1).transpose(1, 2) # (B, H, T, Dh)
    V = V.reshape(B, T, num_heads, -1).transpose(1, 2) # (B, H, T, Dh)
    
    scores = Q @ K.transpose(-2, -1) / Dh ** 0.5
    mask = torch.tril(torch.ones((T, T), dtype=torch.bool))
    scores = scores.masked_fill(~mask[None, None, :, :], -torch.inf)
    W = torch.softmax(scores, dim=-1)

    outputs = W @ V # (B, H, T, Dv)
    outputs = outputs.transpose(1, 2).reshape(B, T, -1) # (B, T, D)
    outputs = outputs @ weights["W_o"].T + weights["b_o"]

    z = x + outputs # (B, T, D)

    z_norm = torch.nn.functional.layer_norm(z, (D, ), eps=eps, weight=weights["gamma2"], bias=weights["beta2"])

    h = z_norm @ weights["W1"].T + weights["b1"]
    h = torch.nn.functional.gelu(h, approximate="tanh")
    h = h @ weights["W2"].T + weights["b2"]

    return h + z
    