import numpy as np

def policy_iteration(P: list, R: list, gamma: float, eval_tol: float = 1e-8, max_iters: int = 200) -> tuple:
    """
    Returns (values, policy): rounded float values and integer actions, one per state.
    """
    P = np.array(P, dtype=np.float64)
    R = np.array(R, dtype=np.float64)
    num_states, num_actions, _ = R.shape

    values = np.zeros(num_states, dtype=np.float64)
    policy = np.zeros(num_states, dtype=np.int64)

    for iteration in range(max_iters):
        # value update
        while True:
            old_values = values.copy()
            for state in range(num_states):
                action = policy[state]
                target_value = 0.0
                for next_state in range(num_states):
                    reward = R[state][action][next_state]
                    p_trans = P[state][action][next_state]
                    target_value += p_trans * (reward + gamma * old_values[next_state])
                values[state] = target_value

            if np.max(np.abs(values - old_values)) < eval_tol:
                break

        old_policy = policy.copy()
        for state in range(num_states):
            best_val = -np.inf
            best_action = 0
            for action in range(num_actions):
                target_value = 0.0
                for next_state in range(num_states):
                    p_trans = P[state][action][next_state]
                    reward = R[state][action][next_state]
                    target_value += p_trans * (reward + gamma * values[next_state])
                if best_val < target_value:
                    best_val = target_value
                    best_action = action
            policy[state] = best_action

        if np.array_equal(old_policy, policy):
            break

    return np.round(values, 4).tolist(), policy.tolist()
                    