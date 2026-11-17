import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d**0.5 
    weights = torch.softmax(scores, dim=-1)

    return weights @ V