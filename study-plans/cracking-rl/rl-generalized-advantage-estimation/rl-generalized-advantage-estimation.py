def gae_advantages(rewards: list, values: list, gamma: float, lam: float, last_value: float = 0.0) -> list:
    """
    Returns the advantage list in timestep order, rounded to four decimals.
    """
    T = len(rewards)
    deltas = []
    for i in range(T):
        next_val = values[i+1] if i < T - 1 else last_value
        delta = rewards[i] + gamma * next_val - values[i]
        deltas.append(delta)

    advantages = [0.0] * (T+1)
    for i in range(T-1, -1, -1):
        advantages[i] = deltas[i] + gamma * lam * advantages[i+1] 

    return advantages[:-1]