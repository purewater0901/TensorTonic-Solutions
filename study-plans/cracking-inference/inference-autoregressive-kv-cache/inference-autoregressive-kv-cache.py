import torch

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Returns (outputs, key_cache, value_cache), float32 tensors in sequence order.
    """
    B, Sq, Dv = query.shape
    Sk = key.shape[1]
    Dk = key.shape[-1]

    scores = query @ key.transpose(-2, -1) / Dk**0.5 # (B, S, S)
    mask = torch.arange(Sq)[:, None] >= torch.arange(Sk)[None, :] # (Sq, Sk)
    scores = scores.masked_fill(~mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1) # (B, Sq, Sk)

    outputs = weights @ value # (B, S, dk)

    return outputs, key, value
