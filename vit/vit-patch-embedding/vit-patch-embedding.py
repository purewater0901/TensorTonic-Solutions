import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    # images # (N H W C)
    N, H, W, C = image.shape
    Hp = H // patch_size
    Wp = W // patch_size

    valid_image = image[:, :Hp*patch_size, :Wp*patch_size, :]
    patches = valid_image.reshape(N, Hp, patch_size, Wp, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5) # (N, Hp, Wp, p, p, C)
    patches = patches.reshape(N, Hp * Wp, -1)
    return patches @ W_proj + bias

    