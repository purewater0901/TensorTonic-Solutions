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

    B, S, D = hidden_states.shape
    num_group = num_query_heads // num_kv_heads
    
    Q = hidden_states @ w_q # (B, S, D)
    K = hidden_states @ w_k # (B, S, Dk)
    V = hidden_states @ w_v # (B, S, Dv)
    Q = Q.reshape(B, S, num_query_heads, -1).transpose(1, 2) # (B, Hq, S, Dq)
    Q = Q.reshape(B, num_kv_heads, num_group, S, -1) # (B, Hkv, G, S, Dq)
    K = K.reshape(B, S, num_kv_heads, -1).transpose(1, 2) # (B, Hkv, S, Dkv)
    V = V.reshape(B, S, num_kv_heads, -1).transpose(1, 2) # (B, Hkv, S, Dkv)
    K = K.unsqueeze(2) # (B, Hkv, 1, S, Dkv)
    V = V.unsqueeze(2) # (B, Hkv, 1, S, Dkv)

    scores = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5 # (B, Hkv, G, S, S)
    if causal:
        mask = torch.tril(torch.ones((S, S), dtype=torch.bool))
        scores = scores.masked_fill(~mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1)

    outputs = weights @ V # (B, Hkv, G, S, Dkv)
    outputs = outputs.reshape(B, -1, S, V.shape[-1]) # (B, Hq, S, Dkv)
    outputs = outputs.transpose(1, 2).reshape(B, S, D)

    return outputs @ w_o
    

        
