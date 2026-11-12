import math
import numpy as np

def softmax_action_probs(Q_values: list, tau: float) -> list:
    """
    Returns the action-probability list, rounded to four decimals.
    """
    Q_values = np.asarray(Q_values)
    max_vals = np.max(Q_values)
    exp_ = np.exp((Q_values - max_vals)/tau)
    pi = exp_ / np.sum(exp_, keepdims=True)
    return pi.tolist()