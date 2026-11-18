import torch

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns causal attention outputs.
    """
    # (B, T, D)
    B, T, D = Q.shape

    scores = Q @ K.transpose(-2, -1) / D ** 0.5
    mask = torch.tril(torch.ones((T, T), dtype=torch.bool))
    scores = scores.masked_fill(~mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1)

    return weights @ V