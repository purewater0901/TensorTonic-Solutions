import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    B, Hq, T, Hd = q.shape
    Hkv = k.shape[1]

    G = Hq // Hkv

    q = q.reshape(B, Hkv, G, T, Hd)
    k = k[:, :, None] # (B, Hkv, 1, T, Hd)
    v = v[:, :, None] # (B, Hkv, 1,  T, Hd)

    scores = q @ k.transpose(-2, -1) / Hd ** 0.5
    mask = torch.tril(torch.ones((T, T), dtype=bool))
    scores = scores.masked_fill(~mask, -torch.inf)

    weights = torch.softmax(scores, dim=-1)
    outputs = weights @ v # (B, Hkv, G, T, Hd)

    return outputs.reshape(B, -1, T, Hd)
    

    
    
