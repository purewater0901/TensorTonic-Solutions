import torch

def softmax_with_sinks(scores: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """
    Returns attention weights with the same shape as scores.
    """
    # scores (H, Tq, Tk)
    # sinks (H)
    max_logits = torch.max(scores, dim=-1, keepdim=True).values
    shifted_logits = scores - max_logits
    exp_ = torch.exp(shifted_logits) # (H, Tq, Tk)
    outputs = exp_ / (torch.sum(exp_, dim=-1, keepdim=True) + torch.exp(sinks[:, None, None] - max_logits))
    return outputs