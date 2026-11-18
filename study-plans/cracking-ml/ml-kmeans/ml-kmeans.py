import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    def find_min_idx(x, centroids):
        # x (B, D)
        # centroids (K, D)
        diff = x[:, None, :] - centroids[None, :, :] # (B, K, D)
        dist = np.linalg.norm(diff, axis=-1) # (B, K)
        min_ids = np.argmin(dist, axis=-1) # (B,)
        return min_ids

    X = np.array(X, dtype=np.float64)
    rng = np.random.RandomState(seed)
    B, D = X.shape
    
    init_ids = rng.choice(B, size=(k, ), replace=False)
    centroids = X[init_ids]

    for _ in range(max_iters):
        old_centroids = centroids.copy()
        min_ids = find_min_idx(X, old_centroids) # (B, )

        for id in range(k):
            mask = id == min_ids
            if not mask.any():
                continue
            target_points = X[mask] # (M, D)
            centroids[id] = np.mean(target_points, axis=0)

        if np.allclose(old_centroids, centroids):
            break

    return find_min_idx(X, centroids), centroids
    

   