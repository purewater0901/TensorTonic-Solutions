import numpy as np

def expected_sarsa_update(Q: list, transitions: list, policy: list, alpha: float, gamma: float) -> list:
    """
    Returns a new updated Q-table as a nested list, rounded to four decimals.
    """
    Q_array = np.array(Q, dtype=np.float64)
    S, A = Q_array.shape

    Q_res = Q_array.copy()
    for transition in transitions:
        Q_old = Q_res.copy()
        state = transition[0]
        action = transition[1]
        reward = transition[2]
        next_state = transition[3]
            
        target = reward
        for next_action in range(A):
            target += gamma * policy[next_state][next_action] * Q_old[next_state][next_action]
        Q_res[state][action] += alpha * (target - Q_res[state][action])

    return np.round(Q_res, 4).tolist()
            