import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    def rotate(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin # (..., D//2)
        x_rot_odd = x_even * sin + x_odd * cos # (..., D//2)
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        x_rot = x_rot.reshape(x.shape)
        return x_rot.to(x.dtype)

    if positions.ndim == 1:
        position_values = positions.float().reshape(1, 1, -1, 1)
    else:
        position_values = positions.float().reshape(positions.shape[0], 1, positions.shape[1], 1)

    angles = position_values * inv_freq.float().reshape(1, 1, 1, -1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    q_rotated = rotate(q, cos, sin)
    k_rotated = rotate(k, cos, sin)

    return {"q_rotated": q_rotated, "k_rotated": k_rotated}

    
