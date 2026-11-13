import numpy as np

def q_learning_update(Q: list, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as Q.
    """
    # Write code here
    target = r + gamma * max(Q[s_next])
    Q[s][a] += alpha * (target - Q[s][a])

    return np.array(Q)