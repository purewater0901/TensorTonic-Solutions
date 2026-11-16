def reinforce_loss(log_probs: list, returns: list) -> float:
    """
    Returns the REINFORCE loss as a float rounded to four decimals.
    """
    T = len(log_probs)
    loss = 0.0
    for t in range(T):
        loss += log_probs[t] * returns[t]

    return -loss / T