import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, Tq, D = q.shape
    Tk = k.shape[1]
    head_dim = D // num_heads
    q = q.reshape(B, Tq, num_heads, head_dim)
    k = k.reshape(B, Tk, num_heads, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / head_dim ** 0.5

    return scores
