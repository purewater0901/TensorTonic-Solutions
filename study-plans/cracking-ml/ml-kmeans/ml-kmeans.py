import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    def find_min_idx(X, centroids):
        dist = X[:, None, :] - centroids[None, :, :] # (B, K, D)
        dist = np.linalg.norm(dist, axis=-1) # (B, K)
        min_ids = np.argmin(dist, axis=-1) # (B, )
        return min_ids # (B, )

    rng = np.random.RandomState(seed)
    X = np.array(X, dtype=np.float64)
    B = len(X)
    init_ids = rng.choice(B, (k, ), replace=False)
    centroids = X[init_ids, :]

    for iteration in range(max_iters):
        min_ids = find_min_idx(X, centroids)

        new_centroids = centroids.copy()
        for i in range(k):
            mask = min_ids == i # (B, )
            
            if not mask.any():
                continue

            target_points = X[mask, :] # (Bm, D)
            new_centroids[i] = np.mean(target_points, axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    min_ids = find_min_idx(X, centroids)
    return min_ids.tolist(), np.round(centroids, 4).tolist()