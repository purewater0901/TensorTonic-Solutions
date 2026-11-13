import torch

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns detached float32 tensors: loss, policy weight/bias, value weight/bias.
    """
    params = []
    for p in (policy_weight, policy_bias, value_weight, value_bias):
        p = p.detach().clone().float().requires_grad_(True)
        params.append(p)

    pw, pb, vw, vb = params
    ha = features @ pw.T + pb # (B, A)
    log_p = torch.log_softmax(ha, dim=-1) # (B, A)

    value = features @ vw + vb # (B, )
    value = torch.tanh(value) # (B, )

    # target_policies # (B, A)
    value_loss = torch.mean((value - target_values)**2)
    action_loss = -torch.mean((target_policies * log_p).sum(dim=-1))
    reg = l2_coefficient * torch.sum((pw**2).sum() + (pb**2).sum() + (vw**2).sum() + (vb**2).sum())

    loss = action_loss + value_loss + reg
    grads = torch.autograd.grad(loss, params)

    with torch.no_grad():
        new_params = []
        for param, grad in zip(params, grads):
            new_param = param - learning_rate * grad
            new_params.append(new_param)

    return loss.detach(), *(p.detach() for p in new_params)
    
