import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
        
    Q = hidden_states @ w_q # (B, T, Dq)
    K = hidden_states @ w_k # (B, T, Dk)
    V = hidden_states @ w_v # (B, T, Dk)
    B, T, _ = Q.shape

    Dq = Q.shape[-1] // num_query_heads
    Dk = K.shape[-1] // num_kv_heads
    G = num_query_heads // num_kv_heads

    Q = Q.reshape(B, T, num_query_heads, -1).transpose(1, 2) # (B, Hq, T, Dq)
    Q = Q.reshape(B, num_kv_heads, G, T, -1) # (B, Hk, G, T,  Dq)
    K = K.reshape(B, T, num_kv_heads, -1).transpose(1, 2) # (B, Hk, T, Dk)
    V = V.reshape(B, T, num_kv_heads, -1).transpose(1, 2) # (B, Hk, T, Dk)
    K = K[:, :, None, :, :] # (B, Hk, 1, T, Dk)
    V = V[:, :, None, :, :] # (B, Hk, 1, T, Dk)

    scores = Q @ K.transpose(-2, -1) / Dq ** 0.5 # (B, Hk, G, T, T)
    if causal:
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool))
        scores = scores.masked_fill(~mask[None, None, None, :, :], -torch.inf)
    weights = torch.softmax(scores, dim=-1) # (B, Hk, G, T, T)

    outputs = weights @ V # (B, Hk, G, T, Dv)
    outputs = outputs.reshape(B, num_query_heads, T, -1) # (B, Hq, T, Dv)
    outputs = outputs.transpose(1, 2).reshape(B, T, -1)

    return outputs @ w_o
    
