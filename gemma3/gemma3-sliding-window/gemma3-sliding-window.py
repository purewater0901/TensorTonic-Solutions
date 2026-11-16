import torch

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Returns a float64 attention output tensor.
    """
    B, H, Tq, Dv = Q.shape
    Tkv = K.shape[-2]
    
    tq = torch.arange(Tq)
    tkv = torch.arange(Tkv)
    
    scores = Q @ K.transpose(-2, -1) / Dv**0.5  # (B, H, Tq, Tq)
    mask = tq[:, None] < tkv[None, :]
    mask = mask | ((tq[:, None] - tkv[None, :]) > window_size)
    scores = scores.masked_fill(mask, -torch.inf)
    weight = torch.softmax(scores, dim=-1)

    return weight @ V