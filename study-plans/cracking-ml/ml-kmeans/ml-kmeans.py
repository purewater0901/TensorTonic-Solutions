import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    def get_min_ids(x, centroids):
        dist = x[:, None, :] - centroids[None, :, :] # (B, k, D)
        dist = np.linalg.norm(dist, axis=-1) # (B, k)
        min_ids = np.argmin(dist, axis=-1) # (B, )
        return min_ids
        
    rng = np.random.RandomState(seed)
    X = np.array(X, dtype=np.float64)
    B, D = X.shape
    centroid_ids = rng.choice(B, size=(k,), replace=False)
    centroids = X[centroid_ids]
    min_ids = get_min_ids(X, centroids)

    for iter in range(max_iters):
        min_ids = get_min_ids(X, centroids)

        new_centroids = centroids.copy()
        for id in range(k):
            mask = min_ids == id
            target_ids = min_ids[mask]
            if len(target_ids) == 0:
                continue
            new_centroids[id] = np.mean(X[mask], axis=0) # (D, )

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    min_ids = get_min_ids(X, centroids)
    return min_ids, centroids
