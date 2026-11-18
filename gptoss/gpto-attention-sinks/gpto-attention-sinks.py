import torch

def softmax_with_sinks(scores: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """
    Returns attention weights with the same shape as scores.
    """
    # scores (H, Tq, Tk)
    # sinks (H)
    H, Tq, Tk = scores.shape
    sinks = sinks[:, None, None].expand(H, Tq, 1)
    logits = torch.cat([scores, sinks], dim=-1)
    
    weights = torch.softmax(logits, dim=-1)
    return weights[..., :-1]