import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T = np.array(T, dtype=np.float64)
    points = np.array(points, dtype=np.float64)
    one_dim = points.ndim == 1
    if one_dim:
        points = points.reshape(1, 3)

    ones = np.ones((points.shape[0], 1))
    points_h = np.concatenate([points, ones], axis=-1) # (B, 3)
    transformed = (T @ points_h.T).T[:, :3]

    return transformed[0] if one_dim else transformed