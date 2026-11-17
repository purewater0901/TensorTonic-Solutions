import torch

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Returns a float64 attention output tensor.
    """
    B, H, Tq, Dh = Q.shape
    Tk = K.shape[-2]
    Dv = Q.shape[-1]

    qt = torch.arange(0, Tq)
    kt = torch.arange(0, Tk)
    scores = Q @ K.transpose(-2, -1) / Dh ** 0.5 # (B, H, Tq, Tk)
    mask = qt[:, None] < kt[None, :] # (Tq, Tk)
    mask |= (qt[:, None] - kt[None, :]) > window_size
    scores = scores.masked_fill(mask[None, None, :, :], -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    
    return weights @ V
