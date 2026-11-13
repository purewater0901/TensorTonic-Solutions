import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    # Write code here
    rng = np.random.default_rng(seed)
    A = len(q_values)

    u = rng.random()
    if u < epsilon:
        return rng.integers(A)

    return np.argmax(q_values)