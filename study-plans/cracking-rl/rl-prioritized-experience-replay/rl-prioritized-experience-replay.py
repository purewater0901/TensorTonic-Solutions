import numpy as np
def per_priorities_and_weights(td_errors: list, alpha: float, beta: float, epsilon: float = 1e-6) -> tuple:
    """
    Returns (probs, is_weights), two float lists rounded to four decimals.
    """
    pi = (np.abs(td_errors) + epsilon)**alpha
    Pi = pi / np.sum(pi)

    N = len(Pi)
    wi = (N * Pi) ** (-beta)
    wi = wi / np.max(wi)

    return np.round(Pi, 4).tolist(), np.round(wi, 4).tolist()