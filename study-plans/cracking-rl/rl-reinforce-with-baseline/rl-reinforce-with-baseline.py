import numpy as np

def reinforce_baseline_loss(log_probs: list, returns: list, baselines: list) -> float:
    """
    Returns the baseline-adjusted loss as a float rounded to four decimals.
    """
    log_probs = np.array(log_probs, dtype=np.float64)
    returns = np.array(returns, dtype=np.float64)
    baselines = np.array(baselines, dtype=np.float64)

    return -np.mean(log_probs * (returns - baselines))