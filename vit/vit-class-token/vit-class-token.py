import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    B = patches.shape[0]
    cls_token = np.broadcast_to(cls_token, (B, 1, cls_token.shape[-1]))
    return np.concatenate([cls_token, patches], axis=1)