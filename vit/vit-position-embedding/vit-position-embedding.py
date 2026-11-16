import numpy as np

def add_position_embedding(patches: np.ndarray,
                           pos_embed: np.ndarray) -> np.ndarray:
    """
    Returns the float64 tokens after adding position embeddings.
    """
    B, T, D = patches.shape
    ts = np.arange(T)
    p_emb = pos_embed[:, ts, :] # (B, T, D)
    return patches + p_emb