import torch

def sparse_moe_forward(
    token_states: torch.Tensor, # (T, D)
    router_logits: torch.Tensor, # (T, E)
    w_in: torch.Tensor, # (E, D, F)
    w_out: torch.Tensor, # (E, F, D)
    top_k: int,
) -> torch.Tensor:
    """
    Returns an output tensor with the same shape as token_states.
    """
    #  F is expert hidden states
    num_tokens, d_model = token_states.shape
    num_experts = router_logits.shape[-1]

    sorted_vals, sorted_idx = torch.sort(router_logits, dim=-1, descending=True, stable=True)
    expert_indices = sorted_idx[:, :top_k] # (T, K)
    routing_weights = torch.softmax(sorted_vals[:, :top_k], dim=-1) # (T, K)

    output = torch.zeros(num_tokens, d_model, dtype=token_states.dtype)

    for expert_id in range(num_experts):
        mask = expert_indices == expert_id # (T, K)
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x = token_states[token_idx] # (M, D)

        hidden = torch.relu(x @ w_in[expert_id])
        expert_out = hidden @ w_out[expert_id]

        weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        output.index_add_(0, token_idx, expert_out * weights)

    return output
