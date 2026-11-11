def policy_gradient_loss(log_probs: list, rewards: list, gamma: float) -> float:
    """
    Returns the REINFORCE loss with a mean-return baseline.
    """
    # Write code here
    total_reward = 0.0
    G = []
    for reward in rewards[::-1]:
        total_reward = reward + gamma * total_reward
        G.append(total_reward)
        
    T = len(log_probs)
    mean_reward = sum(G) / T
    G = G[::-1]
    A = [0.0] * T
    for i in range(T):
        A[i] = G[i] - mean_reward

    loss = 0.0
    for i in range(T):
        loss += log_probs[i] * A[i]
    loss = -loss / T

    return loss