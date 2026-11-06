import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    # q: (B, H, S, D)
    # k: (B, H, S, D)
    # positions: (B, S) -> (B, 1, S, 1)
    # positions: (S) -> (1, 1, S, 1)
    if positions.ndim == 1:
        position_values = positions.float().reshape(1, 1, -1, 1)
    else:
        position_values = positions.float().reshape(positions.shape[0], 1, positions.shape[1], 1)
    
    angles = position_values * inv_freq.float().reshape(1, 1, 1, -1)
    cos = torch.cos(angles) # (1, 1, S, 1)
    sin = torch.sin(angles) # (1, 1, S, 1)

    def rotate(x):
        x_float = x.float()
        even = x_float[..., 0::2] # (B, H, S, D//2)
        odd = x_float[..., 1::2] # (B, H, S, D//2)
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        res = torch.stack((rotated_even, rotated_odd), dim=-1) # (B, H, S, D//2, 2)
        res = res.flatten(-2)
        return res.to(x.dtype)

    return {"q_rotated": rotate(q), "k_rotated": rotate(k)}
