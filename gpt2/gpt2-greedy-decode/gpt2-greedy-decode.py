import torch

def greedy_decode(input_ids: list, step_logits: torch.Tensor) -> list:
    """
    Returns the prompt and greedily generated token IDs.
    """
    res = input_ids.copy()

    for step_logit in step_logits:
        max_id = torch.argmax(step_logit)
        res.append(max_id.item())

    return res