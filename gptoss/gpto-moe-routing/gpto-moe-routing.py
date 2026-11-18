import torch

def moe_route(x: torch.Tensor, gate_W: torch.Tensor, gate_b: torch.Tensor, top_k: int) -> dict:
    """
    Returns top_indices and top_weights in a dictionary.
    """
    G = x @ gate_W + gate_b # (T, E)
    T, E = x.shape
    
    sorted_ids = torch.argsort(G, dim=-1, descending = True)  #(T, E)
    sorted_ids = sorted_ids[:, :top_k]
    sorted_logits = G[torch.arange(T)[:, None], sorted_ids] # (T, K)
    p = torch.softmax(sorted_logits, dim=-1) # (T, K)

    return {
        "top_indices": sorted_ids,
        "top_weights": p
    }